# ria.py
from datetime import datetime
from loguru import logger
from pytz import UTC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from service_scraper.spiders.base import BaseParser
from bs4 import BeautifulSoup
import re
import uuid


class RIAParser(BaseParser):
    """
    Базовый парсер для RIA Новости
    """
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

    def _parse_page_html(self, html: str, stop_datetime=None):
        """
        Специальный парсер для RIA, учитывающий реальную структуру HTML
        """
        soup = BeautifulSoup(html, "html.parser")

        data = []
        text_content = soup.get_text()
        lines = text_content.split('\n')

        current_time = None
        current_views = None

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Ищем строки с временем в формате "09:20", "08:55" и т.д.
            if self._is_time_line(line):
                current_time = line
                continue

            # Ищем количество просмотров (числа типа "438", "796" и т.д.)
            if current_time and self._is_views_line(line):
                current_views = line
                continue

            # Если у нас есть время и это похоже на заголовок новости
            if current_time and len(line) > 20 and not self._is_time_line(line) and not self._is_views_line(
                    line) and not self._is_category_line(line):
                title = line
                url = self._find_url_for_title(soup, title)

                if url and current_time:
                    try:
                        # Создаем полную дату из текущего времени
                        post_date = self._parse_time_string(current_time)
                        if post_date:
                            # Проверяем ограничение по времени
                            if stop_datetime and post_date < stop_datetime:
                                logger.info(f"Достигнуто ограничение по времени: {post_date} < {stop_datetime}")
                                return data, True

                            if url not in self.processed_links:
                                self.processed_links.add(url)

                                # Дополняем относительные URL
                                if url.startswith('/'):
                                    url = 'https://ria.ru' + url

                                from service_scraper.spiders.base import News

                                news_obj = News(
                                    url=url,
                                    title=title,
                                    post_dttm=post_date,
                                    source=self.source,
                                    uuid=str(uuid.uuid4())
                                )
                                data.append(news_obj)
                                logger.debug(f"Статья добавлена: {title[:50]}...")
                    except Exception as e:
                        logger.warning(f"Ошибка при обработке новости '{title}': {e}")

                # Сбрасываем время и просмотры после использования
                current_time = None
                current_views = None

        logger.info(f"Успешно обработано статей: {len(data)}")
        return data, False

    def _is_time_line(self, line: str) -> bool:
        """Проверяем, является ли строка временем в формате 'HH:MM'"""
        pattern = r'^\d{1,2}:\d{2}$'
        return bool(re.match(pattern, line))

    def _is_views_line(self, line: str) -> bool:
        """Проверяем, является ли строка количеством просмотров (просто число)"""
        try:
            int(line)
            return True
        except ValueError:
            return False

    def _is_category_line(self, line: str) -> bool:
        """Проверяем, является ли строка категорией или служебным текстом"""
        categories = ['Экономика', 'Россия', 'В мире', 'Политика', 'Общество', 'Еще', 'Технологии',
                      'Авто', 'Спорт', 'Наука', 'Культура', 'Недвижимость', 'Религия', 'Туризм']
        return line in categories or line.startswith('Связанные Теги') or line == 'Еще'

    def _find_url_for_title(self, soup, title: str) -> str:
        """Ищем URL для заданного заголовка"""
        # Ищем все ссылки и проверяем их текст
        for link in soup.find_all('a', href=True):
            link_text = link.get_text(strip=True)
            if link_text and (title in link_text or link_text in title):
                href = link.get('href')
                # Фильтруем служебные ссылки
                if href and not any(x in href for x in ['#', 'javascript:', 'mailto:', '/tag/', '/person/']):
                    return href
        return None

    def _parse_time_string(self, time_str: str) -> datetime:
        """
        Парсим время в формате "09:20" и создаем дату на сегодня
        """
        try:
            # Создаем дату на сегодня с указанным временем
            now = datetime.now()
            hour, minute = map(int, time_str.split(':'))

            parsed_date = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

            # Если время больше текущего, значит это вчерашняя новость
            if parsed_date > now:
                from datetime import timedelta
                parsed_date = parsed_date - timedelta(days=1)

            # Конвертируем в UTC
            parsed_date = parsed_date.replace(tzinfo=UTC)

            return parsed_date
        except ValueError as e:
            logger.warning(f"Ошибка парсинга времени '{time_str}': {e}")
            return None

    def _parse_date(self, date_tag) -> datetime:
        """
        Заглушка для совместимости с базовым классом
        """
        return None

    def _scroll_to_bottom(self, driver):
        """
        Прокрутка страницы RIA для загрузки дополнительных новостей
        """
        logger.info("Прокручиваем до конца страницы RIA...")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Попробуем найти и нажать кнопку "Еще" если она есть
        try:
            wait = WebDriverWait(driver, 5)
            more_button = wait.until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//button[contains(text(), 'Еще')] | //a[contains(text(), 'Еще')]"))
            )
            more_button.click()
            logger.info("Нажата кнопка 'Еще' для загрузки дополнительных новостей")
        except TimeoutException:
            logger.debug("Кнопка 'Еще' не найдена или не кликабельна")


class RIAEconomicsParser(RIAParser):
    """
    Парсер для экономических новостей RIA
    """

    def __init__(self, pages_url="https://ria.ru/economy/", driver_path: str = None):
        super().__init__(pages_url=pages_url, driver_path=driver_path)


class RIAPoliticsParser(RIAParser):
    """
    Парсер для политических новостей RIA
    """

    def __init__(self, pages_url="https://ria.ru/politics/", driver_path: str = None):
        super().__init__(pages_url=pages_url, driver_path=driver_path)
