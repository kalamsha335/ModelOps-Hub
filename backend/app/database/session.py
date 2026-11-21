from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.exc import OperationalError
import logging
from backend.app.core.config import settings

logger = logging.getLogger(__name__)

# Determine active database URL with fallback
db_url = settings.get_effective_db_url()

try:
    # First attempt PostgreSQL connection with short timeout
    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 2} if "postgresql" in db_url else {}
    )
    with engine.connect() as conn:
        logger.info(f"Connected to primary database: {db_url.split('@')[-1] if '@' in db_url else db_url}")
except Exception as e:
    # Graceful fallback to SQLite for immediate local testing if Postgres container or driver is not active
    logger.warning(f"PostgreSQL not immediately reachable ({type(e).__name__}: {e}). Falling back to local SQLite database.")
    fallback_url = settings.SQLITE_FALLBACK_URL
    engine = create_engine(
        fallback_url,
        connect_args={"check_same_thread": False}
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """
    FastAPI dependency that yields a SQLAlchemy database session per request
    and guarantees safe closure upon request completion.
    """
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
