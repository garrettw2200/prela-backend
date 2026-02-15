"""Pro tier overage tracking and billing service."""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Overage pricing constants (per PRICING_STRATEGY_V2.md)
OVERAGE_PRICING = {
    "traces_per_100k": Decimal("8.00"),
    "users": Decimal("12.00"),
    "ai_hallucination_per_10k": Decimal("5.00"),
    "ai_drift_per_10": Decimal("2.00"),
    "ai_nlp_per_1k": Decimal("3.00"),
    "retention_per_30_days": Decimal("10.00"),
    "ai_debug_per_10": Decimal("1.00"),
}

# Base inclusions for Pro tier
PRO_BASE_LIMITS = {
    "traces": 1_000_000,
    "users": 5,
    "ai_hallucination_checks": 10_000,
    "ai_drift_baselines": 50,
    "ai_nlp_searches": 1_000,
    "retention_days": 90,
    "ai_debug_sessions": 50,
}


class OverageTracker:
    """Track and calculate Pro tier overages for billing."""

    def __init__(self, db_connection=None):
        """Initialize overage tracker.

        Args:
            db_connection: Database connection (optional, for dependency injection)
        """
        self.db = db_connection

    def calculate_overages(
        self,
        subscription_id: str,
        period_start: datetime,
        period_end: datetime,
        usage: Dict[str, int],
    ) -> Dict[str, Decimal]:
        """Calculate overages for a billing period.

        Args:
            subscription_id: Subscription UUID
            period_start: Start of billing period
            period_end: End of billing period
            usage: Dictionary with actual usage:
                {
                    "traces": 1_500_000,
                    "users": 8,
                    "ai_hallucination_checks": 15_000,
                    "ai_drift_baselines": 60,
                    "ai_nlp_searches": 2_000,
                    "retention_days": 90,
                }

        Returns:
            Dictionary with overage costs:
            {
                "traces": Decimal("80.00"),  # 500k over @ $8/100k
                "users": Decimal("36.00"),  # 3 users over @ $12/user
                "ai_hallucination": Decimal("25.00"),  # 50k checks over @ $5/10k
                "ai_drift": Decimal("2.00"),  # 10 baselines over @ $2/10
                "ai_nlp": Decimal("3.00"),  # 1k searches over @ $3/1k
                "retention": Decimal("0.00"),  # No overage
                "total": Decimal("146.00"),
            }
        """
        overages = {}
        total = Decimal("0.00")

        # Calculate trace overages
        if usage.get("traces", 0) > PRO_BASE_LIMITS["traces"]:
            overage_traces = usage["traces"] - PRO_BASE_LIMITS["traces"]
            overage_units = Decimal(overage_traces) / Decimal(100_000)  # Per 100k
            cost = overage_units * OVERAGE_PRICING["traces_per_100k"]
            overages["traces"] = cost.quantize(Decimal("0.01"))
            total += overages["traces"]
            logger.info(
                f"Trace overage: {overage_traces:,} traces = ${overages['traces']}"
            )

        # Calculate user overages
        if usage.get("users", 0) > PRO_BASE_LIMITS["users"]:
            overage_users = usage["users"] - PRO_BASE_LIMITS["users"]
            cost = Decimal(overage_users) * OVERAGE_PRICING["users"]
            overages["users"] = cost.quantize(Decimal("0.01"))
            total += overages["users"]
            logger.info(f"User overage: {overage_users} users = ${overages['users']}")

        # Calculate AI hallucination check overages
        if (
            usage.get("ai_hallucination_checks", 0)
            > PRO_BASE_LIMITS["ai_hallucination_checks"]
        ):
            overage_checks = (
                usage["ai_hallucination_checks"]
                - PRO_BASE_LIMITS["ai_hallucination_checks"]
            )
            overage_units = Decimal(overage_checks) / Decimal(10_000)  # Per 10k
            cost = overage_units * OVERAGE_PRICING["ai_hallucination_per_10k"]
            overages["ai_hallucination"] = cost.quantize(Decimal("0.01"))
            total += overages["ai_hallucination"]
            logger.info(
                f"Hallucination check overage: {overage_checks:,} checks = ${overages['ai_hallucination']}"
            )

        # Calculate drift baseline overages
        if usage.get("ai_drift_baselines", 0) > PRO_BASE_LIMITS["ai_drift_baselines"]:
            overage_baselines = (
                usage["ai_drift_baselines"] - PRO_BASE_LIMITS["ai_drift_baselines"]
            )
            overage_units = Decimal(overage_baselines) / Decimal(10)  # Per 10
            cost = overage_units * OVERAGE_PRICING["ai_drift_per_10"]
            overages["ai_drift"] = cost.quantize(Decimal("0.01"))
            total += overages["ai_drift"]
            logger.info(
                f"Drift baseline overage: {overage_baselines} baselines = ${overages['ai_drift']}"
            )

        # Calculate NLP search overages
        if usage.get("ai_nlp_searches", 0) > PRO_BASE_LIMITS["ai_nlp_searches"]:
            overage_searches = (
                usage["ai_nlp_searches"] - PRO_BASE_LIMITS["ai_nlp_searches"]
            )
            overage_units = Decimal(overage_searches) / Decimal(1_000)  # Per 1k
            cost = overage_units * OVERAGE_PRICING["ai_nlp_per_1k"]
            overages["ai_nlp"] = cost.quantize(Decimal("0.01"))
            total += overages["ai_nlp"]
            logger.info(
                f"NLP search overage: {overage_searches:,} searches = ${overages['ai_nlp']}"
            )

        # Calculate debug session overages
        if usage.get("ai_debug_sessions", 0) > PRO_BASE_LIMITS["ai_debug_sessions"]:
            overage_sessions = (
                usage["ai_debug_sessions"] - PRO_BASE_LIMITS["ai_debug_sessions"]
            )
            overage_units = Decimal(overage_sessions) / Decimal(10)  # Per 10
            cost = overage_units * OVERAGE_PRICING["ai_debug_per_10"]
            overages["ai_debug"] = cost.quantize(Decimal("0.01"))
            total += overages["ai_debug"]
            logger.info(
                f"Debug session overage: {overage_sessions} sessions = ${overages['ai_debug']}"
            )

        # Calculate retention overages
        if usage.get("retention_days", 90) > PRO_BASE_LIMITS["retention_days"]:
            overage_days = usage["retention_days"] - PRO_BASE_LIMITS["retention_days"]
            overage_units = Decimal(overage_days) / Decimal(30)  # Per 30 days
            cost = overage_units * OVERAGE_PRICING["retention_per_30_days"]
            overages["retention"] = cost.quantize(Decimal("0.01"))
            total += overages["retention"]
            logger.info(
                f"Retention overage: {overage_days} days = ${overages['retention']}"
            )

        overages["total"] = total.quantize(Decimal("0.01"))
        logger.info(
            f"Total overages for subscription {subscription_id}: ${overages['total']}"
        )

        return overages

    def record_overage(
        self,
        subscription_id: str,
        period_start: datetime,
        period_end: datetime,
        usage: Dict[str, int],
        overages: Dict[str, Decimal],
    ):
        """Record overage in usage_overages table.

        Args:
            subscription_id: Subscription UUID
            period_start: Start of billing period
            period_end: End of billing period
            usage: Actual usage data
            overages: Calculated overage costs
        """
        if not self.db:
            logger.warning("No database connection, skipping overage recording")
            return

        try:
            query = """
                INSERT INTO usage_overages (
                    subscription_id,
                    period_start,
                    period_end,
                    traces_included,
                    traces_used,
                    traces_overage,
                    traces_overage_cost,
                    users_included,
                    users_active,
                    users_overage,
                    users_overage_cost,
                    ai_hallucination_checks_included,
                    ai_hallucination_checks_used,
                    ai_hallucination_checks_overage,
                    ai_hallucination_checks_cost,
                    ai_drift_baselines_included,
                    ai_drift_baselines_used,
                    ai_drift_baselines_overage,
                    ai_drift_baselines_cost,
                    ai_nlp_searches_included,
                    ai_nlp_searches_used,
                    ai_nlp_searches_overage,
                    ai_nlp_searches_cost,
                    ai_debug_sessions_included,
                    ai_debug_sessions_used,
                    ai_debug_sessions_overage,
                    ai_debug_sessions_cost,
                    retention_days_included,
                    retention_days_used,
                    retention_overage_cost,
                    total_overage_cost
                ) VALUES (
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s
                )
            """

            values = (
                subscription_id,
                period_start,
                period_end,
                # Traces
                PRO_BASE_LIMITS["traces"],
                usage.get("traces", 0),
                max(0, usage.get("traces", 0) - PRO_BASE_LIMITS["traces"]),
                overages.get("traces", Decimal("0.00")),
                # Users
                PRO_BASE_LIMITS["users"],
                usage.get("users", 0),
                max(0, usage.get("users", 0) - PRO_BASE_LIMITS["users"]),
                overages.get("users", Decimal("0.00")),
                # AI hallucination
                PRO_BASE_LIMITS["ai_hallucination_checks"],
                usage.get("ai_hallucination_checks", 0),
                max(
                    0,
                    usage.get("ai_hallucination_checks", 0)
                    - PRO_BASE_LIMITS["ai_hallucination_checks"],
                ),
                overages.get("ai_hallucination", Decimal("0.00")),
                # AI drift
                PRO_BASE_LIMITS["ai_drift_baselines"],
                usage.get("ai_drift_baselines", 0),
                max(
                    0,
                    usage.get("ai_drift_baselines", 0)
                    - PRO_BASE_LIMITS["ai_drift_baselines"],
                ),
                overages.get("ai_drift", Decimal("0.00")),
                # AI NLP
                PRO_BASE_LIMITS["ai_nlp_searches"],
                usage.get("ai_nlp_searches", 0),
                max(
                    0,
                    usage.get("ai_nlp_searches", 0)
                    - PRO_BASE_LIMITS["ai_nlp_searches"],
                ),
                overages.get("ai_nlp", Decimal("0.00")),
                # AI debug
                PRO_BASE_LIMITS["ai_debug_sessions"],
                usage.get("ai_debug_sessions", 0),
                max(
                    0,
                    usage.get("ai_debug_sessions", 0)
                    - PRO_BASE_LIMITS["ai_debug_sessions"],
                ),
                overages.get("ai_debug", Decimal("0.00")),
                # Retention
                PRO_BASE_LIMITS["retention_days"],
                usage.get("retention_days", 90),
                overages.get("retention", Decimal("0.00")),
                # Total
                overages.get("total", Decimal("0.00")),
            )

            cursor = self.db.cursor()
            cursor.execute(query, values)
            self.db.commit()
            cursor.close()

            logger.info(
                f"Recorded overage for subscription {subscription_id}: ${overages.get('total', 0)}"
            )

        except Exception as e:
            logger.error(f"Error recording overage: {e}")
            if self.db:
                self.db.rollback()

    def report_to_stripe(
        self,
        stripe_subscription_id: str,
        overages: Dict[str, Decimal],
    ):
        """Report overages to Stripe by adding invoice line items.

        Adds one-off InvoiceItems to the customer's upcoming invoice for each
        overage category. This approach avoids the complexity of metered
        subscription items and directly charges the calculated overage amounts.

        Args:
            stripe_subscription_id: Stripe subscription ID.
            overages: Calculated overage costs from calculate_overages().
        """
        try:
            import stripe
            from shared.config import settings

            stripe.api_key = settings.stripe_secret_key

            # Get the subscription to find the customer ID
            subscription = stripe.Subscription.retrieve(stripe_subscription_id)
            customer_id = subscription["customer"]

            descriptions = {
                "traces": "Trace overage (beyond 1M included)",
                "users": "Additional team members (beyond 5 included)",
                "ai_hallucination": "AI hallucination check overage",
                "ai_drift": "AI drift baseline overage",
                "ai_nlp": "AI NLP search overage",
                "ai_debug": "AI debug session overage",
                "retention": "Extended retention overage",
            }

            for overage_type, amount in overages.items():
                if overage_type == "total" or amount <= Decimal("0"):
                    continue

                stripe.InvoiceItem.create(
                    customer=customer_id,
                    amount=int(amount * 100),  # Convert dollars to cents
                    currency="usd",
                    description=descriptions.get(
                        overage_type, f"Pro tier overage: {overage_type}"
                    ),
                    subscription=stripe_subscription_id,
                )
                logger.info(
                    f"Added {overage_type} overage of ${amount} to customer {customer_id}"
                )

        except Exception as e:
            logger.error(f"Error reporting overages to Stripe: {e}")
            # Don't raise - allow invoice to proceed without overages


# Helper function to get current period usage
async def get_usage_for_period(
    subscription_id: str,
    period_start: datetime,
    period_end: datetime,
    team_id: Optional[str] = None,
) -> Dict[str, int]:
    """Get usage data for a billing period from various sources.

    Args:
        subscription_id: User ID (used as subscription identifier).
        period_start: Start of period.
        period_end: End of period.
        team_id: Team UUID for counting team members.

    Returns:
        Dictionary with usage counts.
    """
    usage: Dict[str, int] = {
        "traces": 0,
        "users": 1,  # At minimum, the owner
        "ai_hallucination_checks": 0,
        "ai_drift_baselines": 0,
        "ai_nlp_searches": 0,
        "ai_debug_sessions": 0,
        "retention_days": 90,
    }

    # Count team members from PostgreSQL
    if team_id:
        try:
            from shared.database import get_team_member_count

            member_count = await get_team_member_count(team_id)
            usage["users"] = member_count
        except Exception as e:
            logger.warning(f"Failed to get team member count: {e}")

    # Get AI feature usage from Redis
    try:
        from app.middleware.ai_feature_limiter import get_ai_feature_limiter

        limiter = await get_ai_feature_limiter()
        for feature, key in [
            ("hallucination", "ai_hallucination_checks"),
            ("drift", "ai_drift_baselines"),
            ("nlp", "ai_nlp_searches"),
            ("debug", "ai_debug_sessions"),
        ]:
            usage[key] = await limiter.get_usage(subscription_id, feature)
    except Exception as e:
        logger.warning(f"Failed to get AI feature usage from Redis: {e}")

    # Get trace count from subscription's monthly_usage
    try:
        from shared.database import get_subscription_by_user_id

        subscription = await get_subscription_by_user_id(subscription_id)
        if subscription:
            usage["traces"] = subscription.get("monthly_usage", 0)
    except Exception as e:
        logger.warning(f"Failed to get trace usage: {e}")

    return usage
