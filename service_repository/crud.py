from sqlalchemy import and_
from sqlalchemy.orm import Session
from service_repository.models import NewsModel
from datetime import date

def get_corpus(db: Session):
    """Получение корпуса новостей из БД"""
    corpus = db.query(NewsModel).all()
    return corpus

def get_corpus_by_date_range(db: Session, start_date: date, end_date: date):
    """Получение корпуса новостей из БД по диапазону дат"""
    corpus = db.query(NewsModel).filter(and_(NewsModel.post_dttm >= start_date, NewsModel.post_dttm <= end_date))
    return corpus.all()

