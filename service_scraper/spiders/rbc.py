# rbc.py
from datetime import datetime
from loguru import logger
from pytz import UTC

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

    def __init__(self, pages_url="https://www.rbc.ru/economics/?utm_source=topline", driver_path: str = None):
        super().__init__(pages_url=pages_url, driver_path=driver_path)

    def _parse_date(self, date_tag) -> datetime:
        """
        Извлекаем из time datetime-атрибут: <time datetime="2025-01-27T11:00:00+03:00">
        """
        datetime_str = date_tag.get("datetime", "")
        try:
            dt = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S%z")
            return dt
        except ValueError:
            return None

    def _scroll_to_bottom(self, driver):
        """
        На РБК может быть бесконечный скролл или просто динамическая загрузка.
        Для простоты просто листаем до конца.
        """
        logger.info("Прокручиваем до конца страницы RBC...")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")


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

    def __init__(self, pages_url="https://www.rbc.ru/economics/?utm_source=topline", driver_path: str = None):
        super().__init__(pages_url=pages_url, driver_path=driver_path)


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

    def __init__(self, pages_url="https://www.rbc.ru/politics/?utm_source=topline", driver_path: str = None):
        super().__init__(pages_url=pages_url, driver_path=driver_path)