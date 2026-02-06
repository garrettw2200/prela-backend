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
}

# Base inclusions for Pro tier
PRO_BASE_LIMITS = {
    "traces": 1_000_000,
    "users": 5,
    "ai_hallucination_checks": 10_000,
    "ai_drift_baselines": 50,
    "ai_nlp_searches": 1_000,
    "retention_days": 90,
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
        """Report usage to Stripe for metered billing.

        Args:
            stripe_subscription_id: Stripe subscription ID
            overages: Calculated overage costs

        Note: This will be called during invoice creation to add
        usage-based line items to the invoice.
        """
        try:
            import stripe
            from shared.config import settings

            stripe.api_key = settings.stripe_secret_key

            # Report trace overages
            if overages.get("traces", Decimal("0")) > Decimal("0"):
                # Convert cost to units (e.g., $80 = 10 units of 100k traces @ $8 each)
                units = int(
                    overages["traces"] / OVERAGE_PRICING["traces_per_100k"]
                )
                if units > 0:
                    stripe.SubscriptionItem.create_usage_record(
                        settings.stripe_pro_traces_price_id,
                        quantity=units,
                        timestamp=int(datetime.now(timezone.utc).timestamp()),
                    )
                    logger.info(
                        f"Reported {units} trace overage units to Stripe"
                    )

            # Report user overages
            if overages.get("users", Decimal("0")) > Decimal("0"):
                units = int(overages["users"] / OVERAGE_PRICING["users"])
                if units > 0:
                    stripe.SubscriptionItem.create_usage_record(
                        settings.stripe_pro_users_price_id,
                        quantity=units,
                        timestamp=int(datetime.now(timezone.utc).timestamp()),
                    )
                    logger.info(
                        f"Reported {units} user overage units to Stripe"
                    )

            # Similar for AI features...
            # (Implementation simplified for brevity - would add all features)

        except Exception as e:
            logger.error(f"Error reporting to Stripe: {e}")
            # Don't raise - we'll try again on next invoice


# Helper function to get current period usage
def get_usage_for_period(
    subscription_id: str, period_start: datetime, period_end: datetime
) -> Dict[str, int]:
    """Get usage data for a billing period from various sources.

    Args:
        subscription_id: Subscription UUID
        period_start: Start of period
        period_end: End of period

    Returns:
        Dictionary with usage counts
    """
    # This would query:
    # - Traces from usage_records or rate_limiter
    # - Users from subscriptions/team members
    # - AI feature usage from Redis counters

    # Placeholder implementation
    return {
        "traces": 0,
        "users": 5,
        "ai_hallucination_checks": 0,
        "ai_drift_baselines": 0,
        "ai_nlp_searches": 0,
        "retention_days": 90,
    }
