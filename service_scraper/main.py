import pdb
import time
import traceback
from datetime import datetime, timedelta
from typing import List

from loguru import logger

from service_aggregator.list import calculate_and_save_sentiment_scores
from service_repository.dependencies import SessionManager
from service_repository.models import NewsModel
from service_repository.settings import APP_SETTINGS
from service_scraper.spiders import (
    RBCParser, RIAParser, RBCEconomicsParser, RBCPoliticsParser
)

SOURCES = {
    "economic": [
        RBCEconomicsParser("https://www.rbc.ru/economics/?utm_source=topline"),
        RBCParser("https://quote.rbc.ru/?utm_source=topline"),
        RIAParser('https://ria.ru/economy/')
    ],
    "political": [
        RIAParser("https://ria.ru/politics/"),
        RBCPoliticsParser("https://www.rbc.ru/politics/?utm_source=topline")

    ]
}


def get_last_post_dttm(period_days: int) -> datetime:
    return (datetime.now() - timedelta(days=period_days)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )


def save_last_news(news: List[dict], last_post_dttm: datetime, type_source: str):
    news_models = []
    news_emd_models = []
    processed_dttm = datetime.now().replace(tzinfo=None)

    with SessionManager() as db:
        urls = (
            db.query(NewsModel.url)
            .filter(NewsModel.post_dttm >= (last_post_dttm - timedelta(days=10)))
            .all()
        )
        urls = {url[0] for url in urls}

    for row in news:
        if isinstance(row, dict) and row["url"] in urls:
            continue
        news_models.append(
            NewsModel(
                uuid=row["uuid"],
                title=row["title"],
                post_dttm=row["post_dttm"],
                url=row["url"],
                processed_dttm=processed_dttm,
                type=type_source,
                label=row["lable"]
            )
        )
    with SessionManager() as db:
        if news_models:
            logger.info("put NewsModel")
            db.bulk_save_objects(news_models)
            logger.info("put NewsEmbModel")
            db.bulk_save_objects(news_emd_models)
            db.commit()
        else:
            logger.info("Skip DB")


def run_spider():
    while True:
        last_post_dttm = get_last_post_dttm(APP_SETTINGS.SPIDER_PERIOD_DAYS)
        logger.info(f"last_post_dttm {last_post_dttm}")
        for type_source in SOURCES:
            for parser in SOURCES[type_source]:
                logger.info(f"Start {type_source} {parser.pages_url}")
                try:
                    news = parser.parse(stop_datetime=last_post_dttm)
                    if news:
                        news = calculate_and_save_sentiment_scores(news)
                        save_last_news(news, last_post_dttm, type_source)
                        logger.info(f"Done {type_source} {parser.pages_url}")
                    else:
                        logger.error(f"No news for {type_source} {parser.pages_url}")
                except Exception as err:
                    trace = traceback.format_exc()
                    logger.error(f"Unexpected exception {err} trace {trace}")
        logger.info("sleep")
        time.sleep(APP_SETTINGS.SPIDER_WAIT_TIMEOUT_SEC)


if __name__ == "__main__":
    run_spider()

