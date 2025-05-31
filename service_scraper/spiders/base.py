# base.py
import random
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List, Optional

from bs4 import BeautifulSoup
from loguru import logger
from pytz import UTC
from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options


@dataclass
class News:
    url: str
    title: str
    post_dttm: datetime
    source: str
    uuid: str
    processed_dttm: datetime = datetime.now(tz=UTC)

    def __gt__(self, other: "News") -> bool:
        return self.post_dttm > other.post_dttm


class BaseParser(ABC):
    """
    Базовый класс для парсеров.
    """

    ARTICLES_BLOCK = None
    ARTICLES_ATTR = None
    ARTICLE_BLOCK = None
    ARTICLE_ATTR = None
    DATE_BLOCK = None
    DATE_ATTR = None
    TITLE_BLOCK = None
    TITLE_ATTR = None
    URL_BLOCK = None
    URL_ATTR = None

    FAKE_USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Apple Silicon Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
    )

    def __init__(self, pages_url: str, source: str = None, driver_path: str = None):
        self.pages_url = pages_url
        self.source = source if source else self.pages_url
        self.page_parsed = 0
        self.processed_links = set()
        self.driver_path = driver_path  # Путь к ChromeDriver, если не в PATH

    @abstractmethod
    def _parse_date(self, date_tag) -> datetime:
        """Парсим HTML-тег/атрибут, чтобы получить datetime объекта публикации."""
        pass

    @abstractmethod
    def _scroll_to_bottom(self, driver: webdriver.Chrome):
        """Метод для прокрутки / подгрузки следующей порции новостей, если нужно."""
        pass

    def _create_driver(self) -> webdriver.Chrome:
        """Создаём Selenium-драйвер (Chrome) без использования webdriver-manager."""
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # Если нужно окно браузера - убрать
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument(f'user-agent={self.FAKE_USER_AGENT}')

        # Если ChromeDriver не в PATH, укажите путь:
        if self.driver_path:
            service = Service(executable_path=self.driver_path)
        else:
            service = Service()  # Предполагается, что ChromeDriver в PATH

        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.maximize_window()
        return driver

    def parse(
        self,
        max_pages: int = 5,
        start_page: int = 1,
        stop_datetime: Optional[datetime] = None
    ) -> List[dict]:
        """
        Точка запуска парсера.
        :param max_pages: Максимальное количество "прокруток" или страниц.
        :param start_page: Номер стартовой "страницы" (условно).
        :param stop_datetime: Остановиться, если дата статьи стала меньше этого момента.
        :return: список словарей (dict) со статьями
        """
        # Приведём stop_datetime к utc
        stop_datetime = stop_datetime.replace(tzinfo=UTC) if stop_datetime else None

        # Открываем браузер
        driver = self._create_driver()
        logger.info("Драйвер создан")
        logger.info(f"Открыт {self.pages_url}")
        driver.get(self.pages_url)
        time.sleep(2)  # даём странице чуть-чуть времени подгрузиться

        result = []
        self.page_parsed = 0
        page_number = start_page

        try:
            while True:
                logger.info(f"Собираем данные c условной 'страницы' №{page_number} ...")
                # Снимем HTML-код текущего состояния
                page_html = driver.page_source
                data, should_break = self._parse_page_html(page_html, stop_datetime)
                result.extend(data)

                if should_break:
                    # Достигли нужной даты
                    logger.info(f"Достигнуто ограничение по времени {stop_datetime}, выходим из цикла.")
                    break

                page_number += 1
                self.page_parsed += 1

                # Если достигли лимита страниц - выходим
                if max_pages and page_number == start_page + max_pages:
                    break

                # Прокручиваем / подгружаем контент
                self._scroll_to_bottom(driver)
                time.sleep(3)

        finally:
            # После всех действий не забываем закрывать браузер
            driver.quit()

        # Преобразуем в список словарей, т.к. у вас дальше в коде идет asdict
        return [asdict(news_item) for news_item in result]

    def debug_selectors(self, html: str):
        """Отладочный метод для проверки селекторов"""
        soup = BeautifulSoup(html, "html.parser")

        logger.info(f"=== ОТЛАДКА СЕЛЕКТОРОВ ===")
        logger.info(f"ARTICLES_BLOCK: {self.ARTICLES_BLOCK}, ARTICLES_ATTR: {self.ARTICLES_ATTR}")

        articles_block = soup.find(self.ARTICLES_BLOCK, self.ARTICLES_ATTR)
        if articles_block:
            logger.info(f"✓ Блок статей найден")
            articles = articles_block.find_all(self.ARTICLE_BLOCK, class_=self.ARTICLE_ATTR)
            logger.info(f"✓ Найдено статей: {len(articles)}")

            if articles:
                first_article = articles[0]
                logger.info(f"=== ПЕРВАЯ СТАТЬЯ ===")
                logger.info(f"HTML первой статьи: {str(first_article)[:500]}...")

                # Проверяем URL
                if self.URL_BLOCK and self.URL_ATTR:
                    url_tag = first_article.find(self.URL_BLOCK, self.URL_ATTR)
                    logger.info(f"URL_BLOCK найден: {url_tag is not None}")
                    if url_tag:
                        a_tag = url_tag.find('a')
                        logger.info(f"A-тег в URL_BLOCK: {a_tag is not None}")
                        if a_tag:
                            logger.info(f"href: {a_tag.get('href')}")

                # Проверяем дату
                date_tag = first_article.find(self.DATE_BLOCK, self.DATE_ATTR)
                logger.info(f"DATE_BLOCK найден: {date_tag is not None}")
                if date_tag:
                    logger.info(f"Содержимое даты: '{date_tag.get_text(strip=True)}'")

                # Проверяем заголовок
                title_tag = first_article.find(self.TITLE_BLOCK, self.TITLE_ATTR)
                logger.info(f"TITLE_BLOCK найден: {title_tag is not None}")
                if title_tag:
                    logger.info(f"Заголовок: '{title_tag.get_text(strip=True)[:100]}...'")
        else:
            logger.error(f"✗ Блок статей НЕ найден")

    def _parse_page_html(
            self,
            html: str,
            stop_datetime: Optional[datetime] = None
    ) -> (List[News], bool):
        """
        Разбор HTML, взятого из driver.page_source.
        Возвращает кортеж: (список новостей, флаг остановки).
        """
        soup = BeautifulSoup(html, "html.parser")
        articles_block = soup.find(self.ARTICLES_BLOCK, self.ARTICLES_ATTR)

        if not articles_block:
            logger.error(f"Не найден блок статей: {self.ARTICLES_BLOCK} с атрибутами {self.ARTICLES_ATTR}")
            return [], False

        articles = articles_block.find_all(self.ARTICLE_BLOCK, class_=self.ARTICLE_ATTR)
        logger.info(f"Найдено {len(articles)} статей на странице.")

        data = []
        for article in tqdm(articles):
            # URL - исправленная логика
            if self.URL_BLOCK and self.URL_ATTR:
                if self.URL_BLOCK == "a":
                    # Если URL_BLOCK сам является тегом 'a'
                    url_tag = article.find(self.URL_BLOCK, self.URL_ATTR)
                    if not url_tag:
                        logger.debug(f"Не найден URL_BLOCK: {self.URL_BLOCK} с атрибутами {self.URL_ATTR}")
                        continue
                    url = url_tag.get('href')
                else:
                    # Если URL_BLOCK содержит тег 'a'
                    url_tag = article.find(self.URL_BLOCK, self.URL_ATTR)
                    if not url_tag:
                        logger.debug(f"Не найден URL_BLOCK: {self.URL_BLOCK} с атрибутами {self.URL_ATTR}")
                        continue
                    a_tag = url_tag.find('a')
                    if not a_tag:
                        logger.debug(f"Не найден a-тег в URL_BLOCK")
                        continue
                    url = a_tag.get('href')
            else:
                # Если не заданы отдельные блоки - берём первую ссылку
                a_tag = article.find('a')
                if not a_tag:
                    logger.debug(f"Не найден a-тег")
                    continue
                url = a_tag.get('href')

            if not url:
                logger.debug(f"Пустой URL")
                continue

            if url in self.processed_links:
                logger.debug(f"Дубликат URL: {url}")
                continue

            # Дополняем относительные URL
            if url.startswith('/'):
                base_url = 'https://www.rbc.ru'
                url = base_url + url

            self.processed_links.add(url)

            # Дата
            raw_date_tag = article.find(self.DATE_BLOCK, self.DATE_ATTR)
            if not raw_date_tag:
                logger.debug(f"Не найден DATE_BLOCK: {self.DATE_BLOCK} с атрибутами {self.DATE_ATTR}")
                continue

            try:
                post_date = self._parse_date(raw_date_tag)
            except Exception as e:
                logger.warning(f"Не удалось распарсить дату: {e}")
                logger.debug(f"Содержимое тега даты: {raw_date_tag.get_text(strip=True)}")
                continue

            if not post_date:
                logger.debug(f"Дата не получена после парсинга")
                continue

            # Проверяем ограничение по времени
            if stop_datetime and post_date < stop_datetime:
                logger.info(f"Достигнуто ограничение по времени: {post_date} < {stop_datetime}")
                return data, True

            # Заголовок
            title_tag = article.find(self.TITLE_BLOCK, self.TITLE_ATTR)
            if not title_tag:
                logger.debug(f"Не найден TITLE_BLOCK: {self.TITLE_BLOCK} с атрибутами {self.TITLE_ATTR}")
                continue

            title = title_tag.get_text(strip=True)
            if not title:
                logger.debug(f"Пустой заголовок")
                continue

            news_obj = News(
                url=url,
                title=title,
                post_dttm=post_date,
                source=self.source,
                uuid=str(uuid.uuid4())
            )

            data.append(news_obj)
            logger.debug(f"Статья добавлена: {title[:50]}...")

        logger.info(f"Успешно обработано статей: {len(data)}")
        return data, False


