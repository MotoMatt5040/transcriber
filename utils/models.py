import os
from sqlalchemy import create_engine, Column, String, Integer, Date, BigInteger, or_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base
from datetime import date

from dotenv import load_dotenv
load_dotenv()


Base = declarative_base()
engine = create_engine(os.environ['db_uri'])
Session = sessionmaker(bind=engine)
session = Session()


class TranscriptionModel(Base):
    __tablename__ = 'tblTACODetailRecordsClone'

    RecStrID = Column(String(400), primary_key=True)
    ProjectID = Column(String(50))
    TypeFlowStatus = Column(Integer)
    TypeCode = Column(Integer)
    CallDate = Column(Date)
    Transcription = Column(String(8000))
    PCMHome = Column(Integer)


class ActiveProjects(Base):
    __tablename__ = 'tblTACODBS'
    recid = Column(BigInteger, primary_key=True)
    ProjectID = Column(String(50))
    ProcessMode = Column(Integer)
    ProjectStatus = Column(Integer)



process_mode = or_(ActiveProjects.ProcessMode == i for i in [1, 3, 5])
project_status = or_(ActiveProjects.ProjectStatus == i for i in [1, 4])
active_projects = session.query(ActiveProjects).where(process_mode).where(project_status).all()
if not active_projects:
    quit('No projects with required open ends found.')
ore = or_(TranscriptionModel.ProjectID == project.ProjectID for project in active_projects)
result = session.query(TranscriptionModel).where(ore).where(TranscriptionModel.PCMHome > 0).all()


