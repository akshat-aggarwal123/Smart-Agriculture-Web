import os
import sys
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# Add root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Base from your models
from models.farm import Base

# Load settings
from config import settings

config = context.config
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# Set target metadata
target_metadata = Base.metadata