import random
import time
from abc import ABC, abstractstaticmethod, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List, Optional
import uuid
from time import sleep
import pdb
from bs4 import BeautifulSoup
from loguru import logger
from pytz import UTC
from requests import Response, Session
from requests.compat import urljoin
from tqdm import tqdm


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

    ARTICLES_BLOCK = None
    ARTICLES_ATTR = None
    ARTICLE_BLOCK = None
    ARTICLE_ATTR = None
    DATE_BLOCK = None
    DATE_ATTR = None
    TEXT_BLOCK = None
    TEXT_ATTR = None
    TITLE_BLOCK = None
    TITLE_ATTR = None
    URL_BLOCK = None
    URL_ATTR = None

    FAKE_USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36"
    )

    def __init__(self, pages_url: str, source: str = None) -> None:
        self.pages_url = pages_url
        self.source = source if source else self.pages_url
        self.session = Session()
        self.headers = {"User-Agent": self.FAKE_USER_AGENT}
        self.page_parsed = 0
        self.processed_links = set()

    @abstractmethod
    def _parse_date(date) -> datetime:
        """Парсим строчку с созданием времени"""
        raise NotImplemented

    @abstractmethod
    def _scroll_to_bottom(self):
        raise NotImplemented

    @staticmethod
    def _sleep():
        rand = (random.random() + 1) * 1.5
        time.sleep(rand)

    def _do_request(self, url: str, attempt: int = 0) -> Response:
        if attempt >= 5:
            raise Exception("Too many attempts...")
        try:
            response = self.session.get(url, headers=self.headers)
            self._sleep()
        except:
            self._sleep()
            response = self._do_request(url, attempt + 1)
        return response

    def _parse_page(
            self, stop_datetime: Optional[datetime] = None
    ) -> List[News]:
        # page_html = self.driver.page_source
        response = self._do_request(self.pages_url)
        tree = BeautifulSoup(response.content, "html.parser")
        # находим блок со всеми новостями и в нем уже каждый блок новости
        articles_block = tree.find(self.ARTICLES_BLOCK, self.ARTICLES_ATTR)
        # pdb.set_trace()
        if articles_block is not None:
            articles = articles_block.find_all(self.ARTICLE_BLOCK, class_=self.ARTICLE_ATTR)
            if self.URL_BLOCK:
                articles_urls = [
                    article.find(self.URL_BLOCK, self.URL_ATTR).a.get("href")
                    for article in articles
                ]
                # pdb.set_trace()
            else:
                articles_urls = [article.a.get("href") for article in articles]
            logger.info("Scraping articles on page")
            data = []
            for url in tqdm(articles_urls):
                try:
                    if url in self.processed_links:
                        continue
                    else:
                        news = self._parse_article(url, stop_datetime)
                        if news is not None:
                            data.append(news)
                except StopIteration:
                    return data, True
            return data, False
        else:
            logger.error("Не удалось найти блок статей на странице")
            return [], False

    def _parse_article(
        self, article_url: str, stop_datetime: Optional[datetime] = None
    ) -> News:
        """Парсим HTML-страницу с новостью"""
        news_url = urljoin(self.pages_url, article_url)
        response = self._do_request(news_url)
        tree = BeautifulSoup(response.content, "html.parser")
        post_date = None
        for raw_date in tree.findAll(self.DATE_BLOCK, self.DATE_ATTR):
            try:
                post_date = self._parse_date(raw_date)
                break
            except:
                pass

        if post_date is None:
            logger.info("Post date not available, skipping this article")
            return None

        # ограничение по временным рамкам парсинга
        if stop_datetime and post_date and (post_date < stop_datetime):
            logger.info(f"Time stop - {stop_datetime}")
            raise StopIteration

        try:
            title = tree.find(self.TITLE_BLOCK, self.TITLE_ATTR).text
            # pdb.set_trace()
        except AttributeError:
            logger.info("Error: Title block not found")
            return None
        self.processed_links.add(article_url)
        return News(news_url, title, post_date, self.source, uuid.uuid4())

    def parse(
        self,
        max_pages: int = 1,
        start_page: int = 1,
        stop_datetime: Optional[datetime] = None,
    ):
        """Точка запуска парсера"""
        stop_datetime = stop_datetime.replace(tzinfo=UTC) if stop_datetime else None
        assert (
            stop_datetime is not None or max_pages is not None
        ), "Нужно задать ограничения на парсинг"

        result = []
        self.page_parsed = 0
        page_number = start_page
        # self.driver.get(self.pages_url)
        while True:
            logger.info(f"Scraping page №{page_number} ...")
            data, is_breaked = self._parse_page(stop_datetime)
            result.extend(data)
            print(result)
            if is_breaked:
                break
            page_number += 1
            self.page_parsed += 1
            if max_pages and page_number == start_page + max_pages:
                # Мы прошли максимальное число страниц
                break
            else:
                # self._scroll_to_bottom()
                sleep(5)

        return list(map(asdict, result))
