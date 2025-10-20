# db/db.py

import os, importlib
from contextlib import contextmanager
from typing import Optional

from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import sessionmaker, Session


class Database:
    def __init__(
        self,
        db_url: Optional[str] = None,

        driver: str = "postgresql+psycopg2",
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        sslmode: Optional[str] = None, 
        
        models_module: str = "models",
        schema: str = "aura",
        set_search_path: bool = True,
        echo: bool = False,
        pool_size: int = 5,
        max_overflow: int = 10,
    ):
        host      = host      or os.getenv("PGHOST")
        port      = port      or int(os.getenv("PGPORT", "5432"))
        database  = database  or os.getenv("PGDATABASE")
        user      = user      or os.getenv("PGUSER")
        password  = password  or os.getenv("PGPASSWORD")
        sslmode   = sslmode   or os.getenv("PGSSLMODE") or "require"

        raw_url = db_url or os.getenv("DATABASE_URL") or os.getenv("RENDER_DATABASE_URL")

        if host and database and user:
            self._url = URL.create(
                drivername=driver,
                username=user,
                password=password,
                host=host,
                port=port,
                database=database,
                query={"sslmode": sslmode} if sslmode else {},
            )
        elif raw_url:
            self._url = self._prepare_url_from_string(raw_url, default_sslmode=sslmode)
        else:
            raise RuntimeError(
                "Config DB incompleta: suministra host/port/db/user/pass o bien DATABASE_URL."
            )

        self.engine = create_engine(
            self._url,
            echo=echo,
            pool_pre_ping=True,
            pool_size=pool_size,
            max_overflow=max_overflow,
            future=True,
        )

        self.schema = schema
        if set_search_path and self.schema:
            @event.listens_for(self.engine, "connect")
            def _set_search_path(dbapi_conn, conn_record):
                with dbapi_conn.cursor() as cur:
                    cur.execute(f'SET search_path TO "{self.schema}", public;')

        models = importlib.import_module(models_module)
        try:
            self.Base = getattr(models, "Base")
        except AttributeError as e:
            raise RuntimeError(
                f"El módulo '{models_module}' debe exportar 'Base' (Declarative Base)."
            ) from e

        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)

    @staticmethod
    def _prepare_url_from_string(raw_url: str, default_sslmode: Optional[str]) -> URL:
        url = make_url(raw_url)
        q = dict(url.query) if url.query else {}
        if "sslmode" not in q and default_sslmode:
            q["sslmode"] = default_sslmode
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

    def get_session(self) -> Session:
        return self.SessionLocal()
