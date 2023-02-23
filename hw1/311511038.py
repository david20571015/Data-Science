import asyncio
from datetime import datetime
from datetime import timedelta
from functools import reduce
import json
import operator
from pathlib import Path

from bs4 import BeautifulSoup
import click
import requests
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

ALL_ARTICLE_FILE = 'all_article.jsonl'
ALL_POPULAR_FILE = 'all_popular.jsonl'

HEADERS = {
    'user-agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/110.0.0.0 '
                   'Safari/537.36'),
    'cookie': 'over18=1',
}


@click.group()
def cli():
    pass


@cli.command(short_help='Crawl all articles in 2022 from ptt')
@click.option('--use-async', '-a', is_flag=True, help='Use async to crawl')
def crawl(use_async):
    """Crawl all articles in 2022 from ptt.

    Export two json files: all_article.jsonl and all_popular.jsonl.
    all_article.jsonl contains all articles in 2022. all_popular.jsonl
    contains all popular articles in 2022. Both files are in jsonl format:
    `{"date": {date}, "title": {title}, "url": {url}}`.
    """
    billboard = 'Beauty'
    indexes = range(3647, 3955 + 1)
    year = 2022

    host = 'https://www.ptt.cc'

    def parse_article_elememt(article_element):
        title_element = article_element.select_one('div.title a')
        if title_element is None:
            return None, False
        title, url = title_element.text, title_element.get('href')

        meta_element = article_element.select_one('div.meta')
        date = meta_element.select_one('div.date').text.strip()
        date = datetime.strptime(date, '%m/%d').strftime('%m%d')

        is_popular = article_element.select_one('div.nrec').text == '爆'

        return {'date': date, 'title': title, 'url': host + url}, is_popular

    start_date = datetime.strptime(f'{year}0101 00:00:00', '%Y%m%d %H:%M:%S')
    end_date = datetime.strptime(f'{year}1231 23:59:59', '%Y%m%d %H:%M:%S')

    def check_available_date(url: str) -> bool:
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, 'html.parser')
        date_element = soup.select('span.article-meta-value')[-1]
        date = datetime.strptime(date_element.text, '%c')
        return start_date <= date <= end_date

    def crawl_one_page(url: str):
        article_infos = []
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, 'html.parser')
        article_elements = soup.select('div.r-ent')
        for article_element in article_elements:
            article_info, is_popular = parse_article_elememt(article_element)

            if article_info is None:
                continue
            if (article_info['title'].startswith('[公告]') or
                    article_info['title'].startswith('Fw: [公告]')):
                continue
            if (article_info['date'] in ['0101', '1231'] and
                    not check_available_date(article_info['url'])):
                continue

            article_infos.append((article_info, is_popular))

        return article_infos

    def dump_article_infos(article_infos: list[tuple[dict, bool]]):
        with open(
                ALL_ARTICLE_FILE,
                'w',
                encoding='utf-8',
        ) as all_artitle_file, open(
                ALL_POPULAR_FILE,
                'w',
                encoding='utf-8',
        ) as all_popular_file:
            for article_info, is_popular in article_infos:
                print(json.dumps(article_info, ensure_ascii=False),
                      sep='\n',
                      file=all_artitle_file)

                if is_popular:
                    print(json.dumps(article_info, ensure_ascii=False),
                          sep='\n',
                          file=all_popular_file)

    def sync_crawl(urls):
        results = [crawl_one_page(url) for url in tqdm(urls)]

        article_infos = reduce(operator.add, results, [])
        dump_article_infos(article_infos)

    async def async_crawl(urls):
        results = await tqdm_asyncio.gather(
            *[asyncio.to_thread(crawl_one_page, url) for url in urls])

        article_infos = reduce(operator.add, results, [])
        dump_article_infos(article_infos)

    urls = [f'{host}/bbs/{billboard}/index{index}.html' for index in indexes]
    if use_async:
        asyncio.run(async_crawl(urls))
    else:
        sync_crawl(urls)


def _check_file_exists(filename: str):
    if not Path(filename).exists():
        raise FileNotFoundError(
            f'File {filename} does not exist. '
            f'Please run `python {Path(__file__).name} crawl` first.')


def parse_date(date: str) -> datetime:
    return datetime.strptime(date, '%m%d')


@cli.command(short_help='Compute the top 10 user of push/boo')
@click.argument('start-date', type=parse_date)
@click.argument('end-date', type=parse_date)
def push(start_date: datetime, end_date: datetime):
    """Compute the top 10 user of push/boo from START_DATE to END_DATE.

    \b
    Args:
        start_date (datetime): Start date.
        end_date (datetime): End date.
    """
    _check_file_exists(ALL_ARTICLE_FILE)

    with open(ALL_ARTICLE_FILE, 'r', encoding='utf-8') as all_article_file:
        article_infos = [json.loads(line) for line in all_article_file]

    article_infos = filter(
        lambda x: start_date <= parse_date(x['date']) <= end_date,
        article_infos)

    urls = [article_info['url'] for article_info in article_infos]

    def crawl_one_page(url: str):
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, 'html.parser')
        push_elements = soup.select('div.push')
        return [push_element.text for push_element in push_elements]


@cli.command(short_help='Find popular articles and its image urls')
@click.argument('start-date', type=parse_date)
@click.argument('end-date', type=parse_date)
def popular(start_date: datetime, end_date: datetime):
    """Find popular articles and its image urls from START_DATE to END_DATE.

    \b
    Args:
        start_date (datetime): Start date.
        end_date (datetime): End date.
    """
    _check_file_exists(ALL_POPULAR_FILE)


@cli.command(short_help='Find the images of the article contains the keyword')
@click.argument('keyword', type=str)
@click.argument('start-date', type=parse_date)
@click.argument('end-date', type=parse_date)
def keyword(keyword: str, start_date: datetime, end_date: datetime):
    """Find the images of the article contains the KEYWORD from START_DATE to
    END_DATE.

    \b
    Args:
        keyword (str): Keyword.
        start_date (datetime): Start date.
        end_date (datetime): End date.
    """
    _check_file_exists(ALL_ARTICLE_FILE)


if __name__ == '__main__':
    cli()
