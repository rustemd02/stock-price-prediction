# ria.py
from datetime import datetime
from loguru import logger
from pytz import UTC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from service_scraper.spiders.base import BaseParser


class RIAParser(BaseParser):
    ARTICLES_BLOCK = "div"
    ARTICLES_ATTR = {"class": "list list-tags"}
    ARTICLE_BLOCK = "div"
    ARTICLE_ATTR = "list-item"
    DATE_BLOCK = "div"
    DATE_ATTR = {"class": "article__info-date"}
    TITLE_BLOCK = "h1"
    TITLE_ATTR = {"class": "article__second-title"}
    URL_BLOCK = "div"
    URL_ATTR = {"class": "list-item__content"}

    def __init__(self, pages_url="https://ria.ru/economy/", driver_path: str = None):
        super().__init__(pages_url=pages_url, driver_path=driver_path)

    def _parse_date(self, date_tag) -> datetime:
        """
        date_tag: <div class="article__info-date"><a>11:00 27.01.2025</a></div>
        Превращаем в datetime с учетом часового пояса UTC (или без).
        """
        # Сам текст внутри <a>
        a_tag = date_tag.find('a')
        if not a_tag:
            return None
        datetime_str = a_tag.get_text(strip=True)
        # Пример: "11:00 27.01.2025"
        try:
            time_part, date_part = datetime_str.split()
            day, month, year = map(int, date_part.split('.'))
            hour, minute = map(int, time_part.split(':'))
            dt = datetime(year, month, day, hour, minute)
            return dt.replace(tzinfo=UTC)
        except (ValueError, IndexError):
            return None

    def _scroll_to_bottom(self, driver):
        """
        Для RIA пробуем кликнуть на кнопку "Ещё", если она есть.
        Если нет - просто скроллим вниз.
        """
        try:
            load_more_selector = ".list-more.color-btn-second-hover"
            WebDriverWait(driver, 3).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, load_more_selector))
            )
            load_more_button = driver.find_element(By.CSS_SELECTOR, load_more_selector)
            driver.execute_script("arguments[0].scrollIntoView(true);", load_more_button)
            WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, load_more_selector))
            )
            load_more_button.click()
            logger.info("Нажата кнопка 'Ещё' на RIA.")
        except TimeoutException:
            # Если кнопка не появилась, просто скроллим на всякий случай
            logger.info("Кнопка 'Ещё' не найдена, прокручиваем вниз.")
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")