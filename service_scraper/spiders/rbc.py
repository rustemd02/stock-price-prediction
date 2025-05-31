import locale
from datetime import datetime
from loguru import logger
from pytz import UTC
from bs4 import BeautifulSoup

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
        Извлекаем из time datetime-атрибут:
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
    """
    Парсер для экономических новостей RBC.
    Основан на анализе реальной структуры HTML страницы.
    """
    # Основной контейнер со всеми новостями
    ARTICLES_BLOCK = "body"
    ARTICLES_ATTR = None

    # Каждая отдельная новость - это просто абзац с текстом
    # Используем упрощенный подход: ищем все ссылки в контенте
    ARTICLE_BLOCK = "a"
    ARTICLE_ATTR = None

    # Для простого HTML формата RBC экономики
    DATE_BLOCK = None
    DATE_ATTR = None
    TITLE_BLOCK = None
    TITLE_ATTR = None
    URL_BLOCK = None
    URL_ATTR = None

    def _parse_page_html(self, html: str, stop_datetime=None):
        """
        Специальный парсер для RBC Economics, учитывающий реальную структуру HTML
        """
        soup = BeautifulSoup(html, "html.parser")

        # Ищем все новости по паттерну: дата + заголовок
        data = []

        # Находим все ссылки, которые могут быть новостями
        # На основе анализа HTML ищем паттерн: дата затем ссылка
        text_content = soup.get_text()
        lines = text_content.split('\n')

        current_date = None

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Ищем строки с датами в формате "30 мая, 22:58"
            if self._is_date_line(line):
                current_date = line
                continue

            # Если следующая строка после даты - это заголовок новости
            if current_date and len(line) > 20 and not self._is_date_line(line):
                # Ищем ссылку для этого заголовка
                title = line
                url = self._find_url_for_title(soup, title)

                if url and current_date:
                    try:
                        post_date = self._parse_date_string(current_date)
                        if post_date:
                            # Проверяем ограничение по времени
                            if stop_datetime and post_date < stop_datetime:
                                logger.info(f"Достигнуто ограничение по времени: {post_date} < {stop_datetime}")
                                return data, True

                            if url not in self.processed_links:
                                self.processed_links.add(url)

                                # Дополняем относительные URL
                                if url.startswith('/'):
                                    url = 'https://www.rbc.ru' + url

                                from service_scraper.spiders.base import News
                                import uuid

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

                current_date = None  # Сбрасываем дату после использования

        logger.info(f"Успешно обработано статей: {len(data)}")
        return data, False

    def _is_date_line(self, line: str) -> bool:
        """Проверяем, является ли строка датой в формате '30 мая, 22:58'"""
        import re
        pattern = r'\d{1,2}\s+(янв|февр|мар|апр|мая|июн|июл|авг|сент|окт|нояб|дек)[а-я]*,?\s+\d{1,2}:\d{2}'
        return bool(re.search(pattern, line, re.IGNORECASE))

    def _find_url_for_title(self, soup, title: str) -> str:
        """Ищем URL для заданного заголовка"""
        # Ищем все ссылки и проверяем их текст
        for link in soup.find_all('a', href=True):
            link_text = link.get_text(strip=True)
            if link_text and title in link_text:
                return link.get('href')
        return None

    def _parse_date_string(self, date_str: str) -> datetime:
        """
        Парсим дату в формате "30 мая, 22:58"
        """
        # Словарь для замены русских месяцев
        month_mapping = {
            'янв': 'Jan', 'февр': 'Feb', 'мар': 'Mar', 'апр': 'Apr',
            'мая': 'May', 'июн': 'Jun', 'июл': 'Jul', 'авг': 'Aug',
            'сент': 'Sep', 'окт': 'Oct', 'нояб': 'Nov', 'дек': 'Dec'
        }

        try:
            # Заменяем русский месяц на английский
            for rus_month, eng_month in month_mapping.items():
                if rus_month in date_str:
                    date_str = date_str.replace(rus_month, eng_month)
                    break

            # Парсим дату
            parsed_date = datetime.strptime(date_str, "%d %b, %H:%M").replace(year=datetime.now().year)

            # Конвертируем в UTC
            parsed_date = parsed_date.replace(tzinfo=UTC)

            return parsed_date
        except ValueError as e:
            logger.warning(f"Ошибка парсинга даты '{date_str}': {e}")
            return None

    def _parse_date(self, date_tag) -> datetime:
        """Заглушка для совместимости с базовым классом"""
        return None

    def __init__(self, pages_url="https://www.rbc.ru/economics/?utm_source=topline", driver_path: str = None):
        super().__init__(pages_url=pages_url, driver_path=driver_path)


class RBCPoliticsParser(RBCEconomicsParser):
    """
    Политический раздел RBC имеет такую же структуру, как экономический
    """

    def __init__(self, pages_url="https://www.rbc.ru/politics/?utm_source=topline", driver_path: str = None):
        super().__init__(pages_url=pages_url, driver_path=driver_path)
