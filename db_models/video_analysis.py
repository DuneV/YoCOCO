# db/video_analysis.py

from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Integer, TIMESTAMP, func, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from . import Base

class VideoAnalysis(Base):
    __tablename__ = "video_analyses"
    __table_args__ = ({"schema": "aura"},)

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid()  # o func.uuid_generate_v4()
    )

    video_name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    frame_fps: Mapped[Optional[int]] = mapped_column(Integer)
    duration_sec: Mapped[Optional[int]] = mapped_column(Integer)
    frame_count: Mapped[Optional[int]] = mapped_column(Integer)

    yolo_objects: Mapped[List[Dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    arms_pose:   Mapped[List[Dict[str, Any]]]   = mapped_column(JSONB, nullable=False)
    summary:     Mapped[Dict[str, Any]]         = mapped_column(JSONB, nullable=False, server_default="{}")
    meta:        Mapped[Dict[str, Any]]         = mapped_column(JSONB, nullable=False, server_default="{}")

    created_at: Mapped[Any] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())

# Index("idx_va_yolo_gin", VideoAnalysis.yolo_objects, postgresql_using="gin", schema="aura")
# Index("idx_va_arms_gin", VideoAnalysis.arms_pose,   postgresql_using="gin", schema="aura")
# Index("idx_va_summary_gin", VideoAnalysis.summary,  postgresql_using="gin", schema="aura")
# Index("idx_va_meta_gin", VideoAnalysis.meta,        postgresql_using="gin", schema="aura")
