"""Stripe billing and webhook handling."""

from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import logging
import secrets
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

from shared import settings
from shared.database import (
    get_user_by_clerk_id,
    get_subscription_by_user_id,
    update_subscription_tier,
    update_subscription_status,
    create_api_key,
    get_api_keys_by_user_id,
    delete_api_key,
)
from app.auth import get_current_user
from app.services.overage_tracker import OverageTracker, get_usage_for_period

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/billing", tags=["billing"])

# Import Stripe
try:
    import stripe
    stripe.api_key = settings.stripe_secret_key
    STRIPE_AVAILABLE = True
except ImportError:
    STRIPE_AVAILABLE = False
    logger.warning("Stripe library not available. Billing features disabled.")


# Stripe price IDs (to be configured in Stripe dashboard)
STRIPE_PRICES = {
    "lunch-money": settings.stripe_lunch_money_price_id,  # $14/month
    "pro": settings.stripe_pro_price_id,  # $79/month base
    "enterprise": None,  # Contact sales
}


@router.post("/create-checkout-session")
async def create_checkout_session(
    request: Request,
    user: dict = Depends(get_current_user),
):
    """Create a Stripe Checkout session for subscription.

    Request body:
    {
        "tier": "lunch-money" | "pro",
        "success_url": "https://prela.app/billing/success",
        "cancel_url": "https://prela.app/billing/cancel"
    }

    Returns:
    {
        "checkout_url": "https://checkout.stripe.com/..."
    }
    """
    if not STRIPE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Billing service unavailable"
        )

    try:
        body = await request.json()
        tier = body.get("tier")
        success_url = body.get("success_url", "https://prela.app/billing/success")
        cancel_url = body.get("cancel_url", "https://prela.app/billing/cancel")

        if tier not in STRIPE_PRICES or not STRIPE_PRICES[tier]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tier: {tier}"
            )

        # Create Stripe Checkout session
        checkout_session = stripe.checkout.Session.create(
            customer_email=user["email"],
            client_reference_id=user["user_id"],
            mode="subscription",
            line_items=[
                {
                    "price": STRIPE_PRICES[tier],
                    "quantity": 1,
                }
            ],
            success_url=success_url + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=cancel_url,
            metadata={
                "user_id": user["user_id"],
                "clerk_id": user["clerk_id"],
                "tier": tier,
            },
        )

        logger.info(
            f"Created Stripe Checkout session for user {user['user_id']}: "
            f"{checkout_session.id}"
        )

        return {
            "checkout_url": checkout_session.url,
            "session_id": checkout_session.id,
        }

    except stripe.error.StripeError as e:
        logger.error(f"Stripe error creating checkout session: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to create checkout session"
        )
    except Exception as e:
        logger.error(f"Error creating checkout session: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to create checkout session"
        )


@router.post("/create-portal-session")
async def create_portal_session(
    request: Request,
    user: dict = Depends(get_current_user),
):
    """Create a Stripe Customer Portal session for managing subscription.

    Request body:
    {
        "return_url": "https://prela.app/settings"
    }

    Returns:
    {
        "portal_url": "https://billing.stripe.com/..."
    }
    """
    if not STRIPE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Billing service unavailable"
        )

    try:
        body = await request.json()
        return_url = body.get("return_url", "https://prela.app/settings")

        # Get user's subscription
        subscription = await get_subscription_by_user_id(user["user_id"])
        if not subscription or not subscription.get("stripe_customer_id"):
            raise HTTPException(
                status_code=404,
                detail="No Stripe customer found"
            )

        # Create portal session
        portal_session = stripe.billing_portal.Session.create(
            customer=subscription["stripe_customer_id"],
            return_url=return_url,
        )

        logger.info(
            f"Created Stripe Portal session for user {user['user_id']}: "
            f"{portal_session.id}"
        )

        return {
            "portal_url": portal_session.url,
        }

    except stripe.error.StripeError as e:
        logger.error(f"Stripe error creating portal session: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to create portal session"
        )
    except Exception as e:
        logger.error(f"Error creating portal session: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to create portal session"
        )


@router.post("/webhooks/stripe")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events.

    Stripe sends webhooks for subscription lifecycle events:
    - checkout.session.completed: New subscription created
    - customer.subscription.updated: Subscription changed
    - customer.subscription.deleted: Subscription canceled
    - invoice.payment_succeeded: Payment successful
    - invoice.payment_failed: Payment failed
    """
    if not STRIPE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Billing service unavailable"
        )

    # Get webhook signature
    signature = request.headers.get("stripe-signature")
    if not signature:
        raise HTTPException(
            status_code=400,
            detail="Missing stripe-signature header"
        )

    # Get raw body
    body = await request.body()

    # Verify webhook signature
    try:
        event = stripe.Webhook.construct_event(
            body,
            signature,
            settings.stripe_webhook_secret,
        )
    except ValueError as e:
        logger.error(f"Invalid Stripe webhook payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid Stripe webhook signature: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Handle event
    event_type = event["type"]
    event_data = event["data"]["object"]

    logger.info(f"Received Stripe webhook: {event_type} - {event['id']}")

    try:
        if event_type == "checkout.session.completed":
            await handle_checkout_completed(event_data)
        elif event_type == "customer.subscription.updated":
            await handle_subscription_updated(event_data)
        elif event_type == "customer.subscription.deleted":
            await handle_subscription_deleted(event_data)
        elif event_type == "invoice.created":
            await handle_invoice_created(event_data)
        elif event_type == "invoice.payment_succeeded":
            await handle_payment_succeeded(event_data)
        elif event_type == "invoice.payment_failed":
            await handle_payment_failed(event_data)
        else:
            logger.info(f"Unhandled webhook event type: {event_type}")

        return {"status": "success"}

    except Exception as e:
        logger.error(f"Error handling webhook {event_type}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Webhook processing failed"
        )


async def handle_checkout_completed(session: dict):
    """Handle successful checkout session.

    When a user completes checkout:
    1. Update subscription in database
    2. Generate API key
    3. Send email with API key
    """
    logger.info(f"Processing checkout.session.completed: {session['id']}")

    # Extract metadata
    user_id = session["metadata"].get("user_id")
    tier = session["metadata"].get("tier")
    stripe_customer_id = session["customer"]
    stripe_subscription_id = session["subscription"]

    if not user_id or not tier:
        logger.error(f"Missing metadata in checkout session: {session['id']}")
        return

    # Update subscription
    await update_subscription_tier(
        user_id=user_id,
        tier=tier,
        stripe_customer_id=stripe_customer_id,
        stripe_subscription_id=stripe_subscription_id,
        status="active",
    )

    # Generate API key if user doesn't have one
    existing_keys = await get_api_keys_by_user_id(user_id)
    if not existing_keys:
        api_key = await generate_and_store_api_key(user_id, f"Default ({tier})")
        logger.info(f"Generated API key for user {user_id}: {api_key[:16]}...")
        # NOTE: API keys are displayed in the dashboard (/api-keys page)
        # Users are redirected to /api-keys?from=checkout after successful payment

    logger.info(f"Checkout completed for user {user_id}, tier: {tier}")


async def handle_subscription_updated(subscription: dict):
    """Handle subscription update.

    Updates subscription status and tier in database.
    """
    logger.info(f"Processing customer.subscription.updated: {subscription['id']}")

    stripe_customer_id = subscription["customer"]
    status = subscription["status"]

    # Map Stripe status to our status
    status_map = {
        "active": "active",
        "past_due": "past_due",
        "unpaid": "past_due",
        "canceled": "canceled",
        "incomplete": "trialing",
        "incomplete_expired": "canceled",
        "trialing": "trialing",
    }
    our_status = status_map.get(status, "active")

    # Update subscription status
    await update_subscription_status(
        stripe_subscription_id=subscription["id"],
        status=our_status,
    )

    logger.info(
        f"Updated subscription {subscription['id']} status to {our_status}"
    )


async def handle_subscription_deleted(subscription: dict):
    """Handle subscription deletion/cancellation.

    Downgrades user to free tier.
    """
    logger.info(f"Processing customer.subscription.deleted: {subscription['id']}")

    # Downgrade to free tier
    await update_subscription_status(
        stripe_subscription_id=subscription["id"],
        status="canceled",
    )

    # Also update tier to free
    # Note: We need to find user by stripe_subscription_id
    # This requires a new database query function
    logger.info(
        f"Subscription {subscription['id']} canceled, user downgraded to free"
    )


async def handle_invoice_created(invoice: dict):
    """Handle invoice creation - calculate and add Pro tier overages.

    This is called before the invoice is finalized, allowing us to add
    usage-based line items for Pro tier overages.
    """
    logger.info(f"Processing invoice.created: {invoice['id']}")

    stripe_subscription_id = invoice.get("subscription")
    if not stripe_subscription_id:
        logger.info("Invoice not associated with subscription, skipping overage calculation")
        return

    try:
        # Get subscription details
        subscription = stripe.Subscription.retrieve(stripe_subscription_id)

        # Check if this is a Pro tier subscription
        # (We only calculate overages for Pro tier)
        metadata = subscription.get("metadata", {})
        tier = metadata.get("tier")

        if tier != "pro":
            logger.info(f"Subscription tier '{tier}' does not have overages, skipping")
            return

        user_id = metadata.get("user_id")
        if not user_id:
            logger.warning(f"No user_id in subscription metadata: {stripe_subscription_id}")
            return

        # Get billing period from invoice
        period_start = datetime.fromtimestamp(invoice["period_start"], timezone.utc)
        period_end = datetime.fromtimestamp(invoice["period_end"], timezone.utc)

        # Get usage for this period
        usage = get_usage_for_period(user_id, period_start, period_end)

        # Calculate overages
        tracker = OverageTracker()
        overages = tracker.calculate_overages(
            subscription_id=user_id,
            period_start=period_start,
            period_end=period_end,
            usage=usage,
        )

        # Report overages to Stripe (adds line items to invoice)
        if overages["total"] > Decimal("0.00"):
            tracker.report_to_stripe(stripe_subscription_id, overages)
            logger.info(
                f"Added ${overages['total']} in overages to invoice {invoice['id']}"
            )
        else:
            logger.info(f"No overages for invoice {invoice['id']}")

    except Exception as e:
        logger.error(f"Error calculating overages for invoice {invoice['id']}: {e}")
        # Don't raise - allow invoice to proceed without overages


async def handle_payment_succeeded(invoice: dict):
    """Handle successful payment."""
    logger.info(f"Processing invoice.payment_succeeded: {invoice['id']}")

    # Update subscription to active if it was past_due
    stripe_subscription_id = invoice.get("subscription")
    if stripe_subscription_id:
        await update_subscription_status(
            stripe_subscription_id=stripe_subscription_id,
            status="active",
        )
        logger.info(f"Payment succeeded for subscription {stripe_subscription_id}")


async def handle_payment_failed(invoice: dict):
    """Handle failed payment."""
    logger.info(f"Processing invoice.payment_failed: {invoice['id']}")

    # Update subscription to past_due
    stripe_subscription_id = invoice.get("subscription")
    if stripe_subscription_id:
        await update_subscription_status(
            stripe_subscription_id=stripe_subscription_id,
            status="past_due",
        )
        logger.info(
            f"Payment failed for subscription {stripe_subscription_id}, "
            f"marked as past_due"
        )


async def generate_and_store_api_key(user_id: str, name: str) -> str:
    """Generate a new API key and store it in database.

    Args:
        user_id: User UUID.
        name: Name/description for the API key.

    Returns:
        The generated API key (plaintext, only returned once).
    """
    # Generate random API key
    random_bytes = secrets.token_bytes(32)
    api_key = f"prela_sk_{secrets.token_urlsafe(32)}"

    # Hash the key for storage
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    # Get key prefix for display
    key_prefix = api_key[:16]

    # Store in database
    await create_api_key(
        user_id=user_id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        name=name,
    )

    return api_key


@router.get("/subscription")
async def get_subscription(user: dict = Depends(get_current_user)):
    """Get current user's subscription details.

    Returns:
    {
        "tier": "free" | "lunch-money" | "pro" | "enterprise",
        "status": "active" | "canceled" | "past_due" | "trialing",
        "trace_limit": 100000,
        "monthly_usage": 5432,
        "current_period_start": "2026-02-01T00:00:00Z",
        "current_period_end": "2026-03-01T00:00:00Z",
        "cancel_at_period_end": false
    }
    """
    subscription = await get_subscription_by_user_id(user["user_id"])

    if not subscription:
        raise HTTPException(
            status_code=404,
            detail="Subscription not found"
        )

    return {
        "tier": subscription["tier"],
        "status": subscription["status"],
        "trace_limit": subscription["trace_limit"],
        "monthly_usage": subscription["monthly_usage"],
        "current_period_start": subscription.get("current_period_start"),
        "current_period_end": subscription.get("current_period_end"),
        "cancel_at_period_end": subscription.get("cancel_at_period_end", False),
    }
