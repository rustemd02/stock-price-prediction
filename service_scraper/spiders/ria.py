from datetime import datetime, timedelta

from loguru import logger
from pytz import UTC
import pdb

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

    @staticmethod
    def _parse_date(date: str) -> datetime:
        datetime_str = date.a.text
        time, day_month_year = datetime_str.split()
        day, month, year = map(int, day_month_year.split('.'))
        formatted_datetime_str = f"{year:04d}-{month:02d}-{day:02d} {time}:00"
        formatted_date = datetime.strptime(formatted_datetime_str, "%Y-%m-%d %H:%M:%S")
        formatted_date = formatted_date.replace(tzinfo=UTC)
        return formatted_date

    def _scroll_to_bottom(self):
        try:
            WebDriverWait(self.driver, 10).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, ".list-more.color-btn-second-hover"))
            )
            load_more_button = self.driver.find_element(By.CSS_SELECTOR, ".list-more.color-btn-second-hover")
            # Прокрутка до элемента
            self.driver.execute_script("arguments[0].scrollIntoView(true);", load_more_button)
            # Ожидание, пока элемент станет кликабельным
            WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".list-more.color-btn-second-hover"))
            )
            load_more_button.click()
        except TimeoutException:
            # Обработка случая, когда элемент так и не стал кликабельным
            print("Button not clickable")
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")


if __name__ == "__main__":
    buhgalteria_parser = RIAParser("https://ria.ru/economy/")

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