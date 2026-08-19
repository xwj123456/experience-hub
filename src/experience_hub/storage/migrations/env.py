"""Alembic environment for the Experience Hub schema."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool
from sqlalchemy.engine import URL

from experience_hub.storage.tables import Base

config = context.config

if (
    config.config_file_name is not None
    and config.attributes.get("configure_logger", True)
):
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def configured_url() -> str | URL:
    structured_url = config.attributes.get("sqlalchemy_url")
    if structured_url is None:
        fallback_url = config.get_main_option("sqlalchemy.url")
        if fallback_url is None:
            raise RuntimeError("Alembic database URL is not configured")
        return fallback_url
    if not isinstance(structured_url, URL):
        raise TypeError("sqlalchemy_url attribute must be a SQLAlchemy URL")
    return structured_url


def run_migrations_offline() -> None:
    context.configure(
        url=configured_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(
        configured_url(),
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
