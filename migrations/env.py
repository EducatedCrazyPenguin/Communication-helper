"""Migrations share the application's instance lock and backup/recovery boundary."""
from pathlib import Path
import os

from alembic import context
from platformdirs import user_data_path

from communication_gym.storage.schema import metadata

config = context.config


def run(connection):
    context.configure(connection=connection, target_metadata=metadata, compare_type=True, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    raise RuntimeError("Offline migrations are unsupported: use an owned GYM_DATA_DIR and online migration backup protection.")
elif config.attributes.get("connection") is not None:
    run(config.attributes["connection"])
else:
    # Open without invoking another Alembic EnvironmentContext: the command's
    # existing context runs under the same owned lock and recovery boundary.
    from communication_gym.storage.repository import Repository

    root = Path(os.environ.get("GYM_DATA_DIR") or user_data_path("CommunicationGym", appauthor=False))
    repo = Repository(root)
    try:
        repo._initialize(migrate=False)
        repo._migrate(runner=run)
        repo.interrupt_jobs()
    finally:
        repo.close()
