"""Human review of findings: one decision, one status update, one audit event.

Transaction boundary. review_finding() owns its transaction: the new ReviewDecision, the
updated Finding.review_status and the AuditEvent are flushed together and committed ONCE.
If anything fails, everything is rolled back and the exception is re-raised, so a decision
can never exist without its audit event (or the reverse). Callers pass a session with no
pending changes.

Semantics:
  - Each call appends a ReviewDecision. History is never edited or removed.
  - Finding.review_status always equals the latest decision.
  - A finding may be reviewed again, including with the same decision; each review is recorded.
  - Concurrent reviews of one finding are not serialized: the last commit wins the status, and
    both decisions remain in the history.
  - A review changes only the review tables. It never deletes, modifies or "fixes" data.

Privacy: audit metadata holds identifiers, categories, statuses and whether a comment exists,
never the comment text. Reviewer names are self-reported (there is no authentication yet) and
are stored as the audit actor.
"""

import logging

from sqlalchemy.orm import Session

from app.db.models import AuditEvent, Finding, ReviewDecision, ReviewStatus
from app.schemas.api import FindingOut, ReviewDecisionOut, ReviewOut, ReviewRequest
from app.services.query_service import NotFoundError

logger = logging.getLogger(__name__)

DEFAULT_ACTOR = "anonymous"


def review_finding(db: Session, finding_id: int, review: ReviewRequest) -> ReviewOut:
    """Record an approve or reject decision for one finding.

    Raises NotFoundError if the finding does not exist (nothing is written). Any other
    failure rolls the transaction back and is re-raised.
    """
    finding = db.get(Finding, finding_id)
    if finding is None:
        raise NotFoundError("finding", finding_id)

    new_status = ReviewStatus(review.decision.value)
    previous_status = finding.review_status

    try:
        decision = ReviewDecision(
            finding_id=finding.id,
            decision=new_status.value,
            comment=review.comment,
            reviewer=review.reviewer,
        )
        db.add(decision)
        finding.review_status = new_status.value
        db.add(
            AuditEvent(
                actor=review.reviewer or DEFAULT_ACTOR,
                event_type=f"review.{new_status.value}",
                entity_type="finding",
                entity_id=finding.id,
                safe_metadata={
                    "dataset_id": finding.dataset_id,
                    "scan_run_id": finding.scan_run_id,
                    "category": finding.category,
                    "severity": finding.severity,
                    "previous_status": previous_status,
                    "new_status": new_status.value,
                    "has_comment": review.comment is not None,
                    "actor_self_reported": True,
                },
            )
        )
        db.flush()  # assigns ids and timestamps; the response is built before the commit
        output = ReviewOut(
            decision=ReviewDecisionOut.model_validate(decision),
            finding=FindingOut.model_validate(finding),
        )
        db.commit()
    except Exception:
        db.rollback()
        raise

    logger.info("Finding reviewed: finding_id=%d decision=%s", finding_id, new_status.value)
    return output