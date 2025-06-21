<%!
import datetime
import logging
from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey
from sqlalchemy.orm import declarative_base
from app.models.base import Base
%>

<%block name="imports">
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from app.models.base import Base
</%block>

<%block name="upgrades">
def upgrade():
    # Example upgrade template
    pass
</%block>

<%block name="downgrades">
def downgrade():
    # Example downgrade template
    pass
</%block>