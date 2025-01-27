"""
此文件负责处理里程碑相关的数据库操作。
主要包含里程碑的创建、更新和检查等功能，用于追踪用户使用助手的情况。
"""

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from onyx.configs.constants import MilestoneRecordType
from onyx.db.models import Milestone
from onyx.db.models import User


# 用户助手使用记录的前缀
USER_ASSISTANT_PREFIX = "user_assistants_used_"
# 多助手使用标记
MULTI_ASSISTANT_USED = "multi_assistant_used"


def create_milestone(
    user: User | None,
    event_type: MilestoneRecordType,
    db_session: Session,
) -> Milestone:
    """
    创建新的里程碑记录。
    
    此函数用于在数据库中创建一个新的里程碑事件，记录用户的特定行为或成就。
    """
    milestone = Milestone(
        event_type=event_type,
        user_id=user.id if user else None,
    )
    db_session.add(milestone)
    db_session.commit()

    return milestone


def create_milestone_if_not_exists(
    user: User | None, event_type: MilestoneRecordType, db_session: Session
) -> tuple[Milestone, bool]:
    """
    检查并创建里程碑记录。
    
    此函数首先检查特定类型的里程碑是否存在，如果不存在则创建新的里程碑。
    返回值中的布尔值表示是否新创建了里程碑。
    """
    # Check if it exists
    milestone = db_session.execute(
        select(Milestone).where(Milestone.event_type == event_type)
    ).scalar_one_or_none()

    if milestone is not None:
        return milestone, False

    # If it doesn't exist, try to create it.
    try:
        milestone = create_milestone(user, event_type, db_session)
        return milestone, True
    except IntegrityError:
        # Another thread or process inserted it in the meantime
        db_session.rollback()
        # Fetch again to return the existing record
        milestone = db_session.execute(
            select(Milestone).where(Milestone.event_type == event_type)
        ).scalar_one()  # Now should exist
        return milestone, False


def update_user_assistant_milestone(
    milestone: Milestone,
    user_id: str | None,
    assistant_id: int,
    db_session: Session,
) -> None:
    """
    更新用户使用助手的里程碑记录。
    
    此函数用于记录用户使用特定助手的情况，并将这些使用记录存储在里程碑的事件追踪器中。
    如果用户已经达到使用多个助手的里程碑，则不再继续追踪。
    """
    event_tracker = milestone.event_tracker
    if event_tracker is None:
        milestone.event_tracker = event_tracker = {}

    if event_tracker.get(MULTI_ASSISTANT_USED):
        # No need to keep tracking and populating if the milestone has already been hit
        return

    user_key = f"{USER_ASSISTANT_PREFIX}{user_id}"

    if event_tracker.get(user_key) is None:
        event_tracker[user_key] = [assistant_id]
    elif assistant_id not in event_tracker[user_key]:
        event_tracker[user_key].append(assistant_id)

    flag_modified(milestone, "event_tracker")
    db_session.commit()


def check_multi_assistant_milestone(
    milestone: Milestone,
    db_session: Session,
) -> tuple[bool, bool]:
    """
    检查用户是否达到使用多个助手的里程碑。
    Returns if the milestone was hit and if it was just hit for the first time
    返回值说明：是否达到里程碑以及是否是首次达到
    
    此函数用于检查用户是否使用了多个不同的助手，并在首次达到该里程碑时进行标记。
    返回两个布尔值，分别表示是否达到里程碑和是否是首次达到。
    """
    event_tracker = milestone.event_tracker
    if event_tracker is None:
        return False, False

    if event_tracker.get(MULTI_ASSISTANT_USED):
        return True, False

    for key, value in event_tracker.items():
        if key.startswith(USER_ASSISTANT_PREFIX) and len(value) > 1:
            event_tracker[MULTI_ASSISTANT_USED] = True
            flag_modified(milestone, "event_tracker")
            db_session.commit()
            return True, True

    return False, False
