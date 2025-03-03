import os
import logging

from sqlalchemy import create_engine, Column, String, Integer, Date, BigInteger, or_, not_, and_, CHAR, Text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base
from utils.logger_config import logger

from dotenv import load_dotenv
load_dotenv()


Base = declarative_base()
engine = create_engine(
    os.environ['db_uri'],
    pool_pre_ping=True,
    connect_args={"timeout": 30}
)
Session = sessionmaker(bind=engine)
session = Session()


class DetailRecords(Base):
    __tablename__ = 'tblTACODetailRecords'
    RecStrID = Column(String(400), primary_key=True)
    ProjectID = Column(String(50))
    Question = Column(String(50))
    SurveyID = Column(CHAR(10))
    TypeFlowStatus = Column(Integer)
    TypeCode = Column(Integer)
    CallDate = Column(Date)
    Response = Column(String(8000))
    Transcription = Column(String(8000))
    PCMHome = Column(Integer)


class ActiveProjects(Base):
    __tablename__ = 'tblTACODBS'
    recid = Column(BigInteger, primary_key=True)
    ProjectID = Column(String(50))
    ProcessMode = Column(Integer)
    ProjectStatus = Column(Integer)


class Questions(Base):
    __tablename__ = 'tblTACOQES'
    recid = Column(BigInteger, primary_key=True)
    ProjectID = Column(String(50))
    OENum = Column(String(50))
    QText = Column(Text)


class ProjectTranscriptionManager:
    def __init__(self, session):
        self.session = session
        self._active_projects = None

    def update_active_projects(self):
        process_mode = or_(ActiveProjects.ProcessMode == i for i in [1, 3, 5])
        project_status = or_(ActiveProjects.ProjectStatus == i for i in [1, 4])
        is_com = not_(ActiveProjects.ProjectID.contains("COM"))
        try:
            self._active_projects = self.session.query(ActiveProjects).where(and_(process_mode, project_status, is_com)).all()
        except OperationalError as e:
            logger.error("OperationalError during project data pull. Retrying...")
            self.session.rollback()  # Rollback in case of connection loss
            self.update_active_projects()  # Retry the update process
        except Exception as e:
            logger.error("Error pulling active projects. Rollback initiated...")
            session.rollback()

    @property
    def active_projects(self):
        return self._active_projects

    def projects_to_transcribe(self, retries: int = 0):
        try:
            self.update_active_projects()
            if not self.active_projects:
                return None

            active_questions_projects_list = or_(Questions.ProjectID == project.ProjectID for project in self.active_projects)
            questions = self.session.query(Questions).where(active_questions_projects_list).all()
            questions = self.questions_dict(questions)

            active_detail_projects_list = or_(DetailRecords.ProjectID == project.ProjectID for project in self.active_projects)

            result = self.session.query(DetailRecords).where(
                and_(
                    active_detail_projects_list,
                    DetailRecords.PCMHome > 0,
                    DetailRecords.TypeFlowStatus == 14,
                    DetailRecords.TypeCode == 0,
                    DetailRecords.Transcription.is_(None)
                )).all()
            return questions, result
        except OperationalError as e:
            logger.error("OperationalError during project data pull. Retrying...")
            self.session.rollback()  # Rollback in case of connection loss
            self.update_active_projects()  # Retry the update process
        except Exception as e:
            logger.error(e)
            logger.warning("Error pulling project data. Rollback initiated...")
            session.rollback()
            if retries < 5:
                logger.warning(f"Retrying... Attempt {retries + 1}")
                self.projects_to_transcribe(retries + 1)
            else:
                logger.critical("Max retries reached. Exiting...")

                quit()
        return []

    def questions_dict(self, questions):
        if not questions:
            logger.error("No questions found")
            return {}
        q = {}

        for question in questions:
            text: list = question.QText.split('\n')
            try:
                q[question.OENum] = {'text': text[0], 'probe': text[4]}
            except IndexError as e:
                logger.warning(f"{e} - Error creating dictionary for question {question.OENum}:\n*****\n{question.QText}\n*****")
                q[question.OENum] = {'text': '', 'probe': ''}
            except Exception as e:
                logger.error(f"{e} - Error creating dictionary for question {question.OENum}: {question.QText}")
                logger.error("Error creating questions dictionary")
        return q
