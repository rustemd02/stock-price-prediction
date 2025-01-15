from datetime import datetime, timedelta

from loguru import logger
from pytz import UTC
import pdb

from service_scraper.spiders.base import BaseParser


class RBCParser(BaseParser):
    ARTICLES_BLOCK = "div"
    ARTICLES_ATTR = {"class": "js-load-container"}
    ARTICLE_BLOCK = "div"
    ARTICLE_ATTR = "q-item js-rm-central-column-item js-load-item"
    DATE_BLOCK = "time"
    DATE_ATTR = {"class": "article__header__date"}
    TITLE_BLOCK = "h1"
    TITLE_ATTR = {"class": "article__header__title-in js-slide-title"}
    URL_BLOCK = "div"
    URL_ATTR = {"class": "q-item__wrap l-col-center-590"}

    @staticmethod
    def _parse_date(date: str) -> datetime:
        try:
            # Извлекаем содержимое атрибута "datetime"
            datetime_str = date.get("datetime")
            # Парсим строку времени
            date_obj = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S%z")
            return date_obj
        except (ValueError, TypeError):
            return None

    def _scroll_to_bottom(self):
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

class RBCEconomicsParser(RBCParser):
    ARTICLES_BLOCK = "div"
    ARTICLES_ATTR = {"class": "l-row js-load-container"}
    ARTICLE_BLOCK = "div"
    ARTICLE_ATTR = "item js-rm-central-column-item item_image-mob js-category-item"
    DATE_BLOCK = "time"
    DATE_ATTR = {"class": "article__header__date"}
    TITLE_BLOCK = "h1"
    TITLE_ATTR = {"class": "article__header__title-in js-slide-title"}
    URL_BLOCK = "div"
    URL_ATTR = {"class": "item__wrap l-col-center"}

    @staticmethod
    def _parse_date(date: str) -> datetime:
        try:
            # Извлекаем содержимое атрибута "datetime"
            datetime_str = date.get("datetime")
            # Парсим строку времени
            date_obj = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S%z")
            return date_obj
        except (ValueError, TypeError):
            return None

class RBCPoliticsParser(RBCParser):
    ARTICLES_BLOCK = "div"
    ARTICLES_ATTR = {"class": "l-row js-load-container"}
    ARTICLE_BLOCK = "div"
    ARTICLE_ATTR = "item js-rm-central-column-item item_image-mob js-category-item"
    DATE_BLOCK = "time"
    DATE_ATTR = {"class": "article__header__date"}
    TITLE_BLOCK = "h1"
    TITLE_ATTR = {"class": "article__header__title-in js-slide-title"}
    URL_BLOCK = "div"
    URL_ATTR = {"class": "item__wrap l-col-center"}

    @staticmethod
    def _parse_date(date: str) -> datetime:
        try:
            # Извлекаем содержимое атрибута "datetime"
            datetime_str = date.get("datetime")
            # Парсим строку времени
            date_obj = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S%z")
            return date_obj
        except (ValueError, TypeError):
            return None

if __name__ == "__main__":
    buhgalteria_parser = RBCPoliticsParser("https://www.rbc.ru/politics/?utm_source=topline")

    last_bd_time = datetime.now(tz=UTC) - timedelta(days=1)
    first_test = buhgalteria_parser.parse(stop_datetime=last_bd_time)
    min_loaded_date = min([article["post_dttm"] for article in first_test])
    status = "✅" if min_loaded_date > last_bd_time else "❌"
    logger.info(f"Тест №1 - остановимся по времени : {status}")

    page_to_parse = 2
    _ = buhgalteria_parser.parse(max_pages=page_to_parse)
    second_test = buhgalteria_parser.page_parsed
    status = "✅" if page_to_parse == second_test else "❌"
    logger.info(f"Тест №2 - остановимся по страницам : {status}")
