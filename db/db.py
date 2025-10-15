# db.py
import os
import importlib
from contextlib import contextmanager
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import sessionmaker, Session


class Database:
    def __init__(
        self,
        db_url: Optional[str] = None,
        models_module: str = "models",  # p.ej. "app.models"
        echo: bool = False,
        pool_size: int = 5,
        max_overflow: int = 10,
    ):
        self._url = self._prepare_url(db_url or os.getenv("DATABASE_URL") or os.getenv("RENDER_DATABASE_URL"))
        if self._url is None:
            raise RuntimeError(
                "No se encontró DATABASE_URL/RENDER_DATABASE_URL ni se pasó db_url."
            )

        self.engine = create_engine(
            self._url,
            echo=echo,
            pool_pre_ping=True,
            pool_size=pool_size,
            max_overflow=max_overflow,
            future=True,
        )

        models = importlib.import_module(models_module)
        try:
            self.Base = getattr(models, "Base")
        except AttributeError as e:
            raise RuntimeError(
                f"El módulo '{models_module}' debe exportar 'Base' (Declarative Base)."
            ) from e

        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)

    @staticmethod
    def _prepare_url(raw_url: Optional[str]) -> Optional[URL]:
        if not raw_url:
            return None
        url = make_url(raw_url)

        # Render normalmente exige SSL. Garantizamos sslmode=require si no está presente.
        q = dict(url.query) if url.query else {}
        if "sslmode" not in q:
            q["sslmode"] = "require"
            url = url.set(query=q)

        return url

    def create_all(self) -> None:
        self.Base.metadata.create_all(self.engine)

    def drop_all(self) -> None:
        self.Base.metadata.drop_all(self.engine)

    def dispose(self) -> None:
        self.engine.dispose()

    def healthcheck(self) -> bool:
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True

    @contextmanager
    def session(self) -> Session:
        db: Session = self.SessionLocal()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    # Útil si usas frameworks que inyectan la sesión
    def get_session(self) -> Session:
        return self.SessionLocal()
