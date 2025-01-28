# run_spider.py
import time
import traceback
from datetime import datetime, timedelta
from typing import List

from loguru import logger
from service_aggregator.list import calculate_and_save_sentiment_scores
from service_repository.dependencies import SessionManager
from service_repository.models import NewsModel
from service_repository.settings import APP_SETTINGS

from service_scraper.spiders.rbc import RBCParser, RBCEconomicsParser, RBCPoliticsParser
from service_scraper.spiders.ria import RIAParser


SOURCES = {
    "economic": [
        RBCEconomicsParser("https://www.rbc.ru/economics/?utm_source=topline", driver_path="/Users/unterlantas/Downloads/Katalon/chromedriver"),
        RBCParser("https://quote.rbc.ru/?utm_source=topline", driver_path="/Users/unterlantas/Downloads/Katalon/chromedriver"),
        RIAParser("https://ria.ru/economy/", driver_path="/Users/unterlantas/Downloads/Katalon/chromedriver")
    ],
    "political": [
        RIAParser("https://ria.ru/politics/", driver_path="/Users/unterlantas/Downloads/Katalon/chromedriver"),
        RBCPoliticsParser("https://www.rbc.ru/politics/?utm_source=topline", driver_path="/Users/unterlantas/Downloads/Katalon/chromedriver")
    ]
}


def get_last_post_dttm(period_days: int) -> datetime:
    return (datetime.now() - timedelta(days=period_days)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )


def save_last_news(news: List[dict], last_post_dttm: datetime, type_source: str):
    """
    Сохраняем полученные новости в БД, если их ещё нет.
    """
    news_models = []
    processed_dttm = datetime.now().replace(tzinfo=None)

    with SessionManager() as db:
        urls_in_db = (
            db.query(NewsModel.url)
            .filter(NewsModel.post_dttm >= (last_post_dttm - timedelta(days=10)))
            .all()
        )
        urls_in_db = {url[0] for url in urls_in_db}

    for row in news:
        if isinstance(row, dict) and row["url"] in urls_in_db:
            continue
        news_models.append(
            NewsModel(
                uuid=row["uuid"],
                title=row["title"],
                post_dttm=row["post_dttm"],
                url=row["url"],
                processed_dttm=processed_dttm,
                type=type_source,
                label=row.get("label") or row.get("lable")  # учитываем возможную опечатку
            )
        )

    with SessionManager() as db:
        if news_models:
            logger.info(f"Добавляем {len(news_models)} новых записей в {type_source}.")
            db.bulk_save_objects(news_models)
            db.commit()
        else:
            logger.info(f"Нет новых записей для {type_source}, пропускаем.")


def run_spider():
    while True:
        last_post_dttm = get_last_post_dttm(APP_SETTINGS.SPIDER_PERIOD_DAYS)
        logger.info(f"last_post_dttm {last_post_dttm}")
        for type_source, parser_list in SOURCES.items():
            for parser in parser_list:
                logger.info(f"Запуск парсинга: {type_source} | {parser.pages_url}")
                try:
                    news = parser.parse(stop_datetime=last_post_dttm)
                    logger.info(f"Получено {len(news)} новостей от {parser.pages_url}")
                    if news:
                        # sentiment
                        news = calculate_and_save_sentiment_scores(news)
                        # save
                        save_last_news(news, last_post_dttm, type_source)
                        logger.info(f"Парсинг завершён: {type_source} | {parser.pages_url}")
                    else:
                        logger.warning(f"Не найдено новостей для: {type_source} | {parser.pages_url}")
                except Exception as err:
                    trace = traceback.format_exc()
                    logger.error(f"Ошибка парсинга {parser.pages_url}: {err}\n{trace}")
        logger.info("Ожидаем следующий цикл...")
        time.sleep(APP_SETTINGS.SPIDER_WAIT_TIMEOUT_SEC)


if __name__ == "__main__":
    run_spider()