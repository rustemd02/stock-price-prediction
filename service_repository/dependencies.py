from sqlalchemy.orm import Session

from service_repository.database import SessionLocal


class SessionManager:
    def __init__(self):
        self.session: Session = SessionLocal()

    def __enter__(self):
        return self.session

    def __exit__(self, exception_type, exception_value, traceback):
        self.session.close()


def get_db() -> Session:
    with SessionManager() as db:
        yield db
