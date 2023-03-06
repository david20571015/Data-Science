import asyncio
from collections import Counter
from datetime import datetime
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
@click.option('--use-async/--no-async', default=True, help='Use async to crawl')
def crawl(use_async: bool):
    """Crawl all articles in 2022 from ptt.

    Export two json files: all_article.jsonl and all_popular.jsonl.
    all_article.jsonl contains all articles in 2022. all_popular.jsonl
    contains all popular articles in 2022. Both files are in jsonl format:
    `{"date": {date}, "title": {title}, "url": {url}}`.
    """
    billboard = 'Beauty'
    indexes = range(3500, 3960)  # TODO: must be computed dynamically
    year = 2022

    host = 'https://www.ptt.cc'

    session = requests.Session()

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
        response = session.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'lxml')
        try:
            date_element = soup.select('span.article-meta-value')[-1]
            date = datetime.strptime(date_element.text, '%c')
        except Exception:
            return False
        return start_date <= date <= end_date

    def crawl_one_page(url: str):
        article_infos = []
        response = session.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'lxml')
        article_elements = soup.select('div.r-ent')
        for article_element in article_elements:
            article_info, is_popular = parse_article_elememt(article_element)

            if article_info is None:
                continue
            if (article_info['title'].startswith('[公告]') or
                    article_info['title'].startswith('Fw: [公告]')):
                continue

            article_infos.append((article_info, is_popular))

        return article_infos

    def dump_article_infos(article_infos: list[tuple[dict, bool]]):
        # remove articles that are not in 2022
        while not article_infos[0][0]['date'] == '0101':
            article_infos.pop(0)
        while not article_infos[-1][0]['date'] == '1231':
            article_infos.pop()

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


def filter_date_interval(article_infos: list, start_date: str, end_date: str):
    start_datetime = parse_date(start_date)
    end_datetime = parse_date(end_date)
    return filter(
        lambda x: start_datetime <= parse_date(x['date']) <= end_datetime,
        article_infos,
    )


def load_article_infos(
    filename: str,
    start_date: str = '0101',
    end_date: str = '1231',
):
    _check_file_exists(filename)

    with open(filename, 'r', encoding='utf-8') as file:
        article_infos = [json.loads(line) for line in file]

    article_infos = filter_date_interval(article_infos, start_date, end_date)
    return article_infos


@cli.command(short_help='Compute the top 10 user of push/boo')
@click.argument('start-date', type=str)
@click.argument('end-date', type=str)
@click.option('--use-async/--no-async', default=True, help='Use async to crawl')
def push(start_date: str, end_date: str, use_async: bool):
    """Compute the top 10 user of like/boo from START_DATE to END_DATE.

    \b
    Args:
        start_date (str): Start date.
        end_date (str): End date.
    """
    article_infos = load_article_infos(ALL_ARTICLE_FILE, start_date, end_date)

    session = requests.Session()

    def crawl_one_page(url: str):
        response = session.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'lxml')
        push_elements = soup.select('div.push')

        like_list, boo_list = [], []
        for push_element in push_elements:
            push_tag = push_element.select_one('span.push-tag').text.strip()
            push_userid = push_element.select_one(
                'span.push-userid').text.strip()
            if push_tag == '推':
                like_list.append(push_userid)
            elif push_tag == '噓':
                boo_list.append(push_userid)

        return {
            'all_like': len(like_list),
            'all_boo': len(boo_list),
            'likes': Counter(like_list),
            'boos': Counter(boo_list),
        }

    def dump_like_boo_info(like_boo_info: dict):
        result = {
            'all_like': like_boo_info['all_like'],
            'all_boo': like_boo_info['all_boo'],
        }

        top_10_likes = sorted(
            like_boo_info['likes'].most_common(),
            key=operator.itemgetter(1, 0),  # sort by count, then by user_id
            reverse=True,
        )[:10]
        for rank, (uid, count) in enumerate(top_10_likes):
            result[f'like {rank+1}'] = {'user_id': uid, 'count': count}

        top_10_boos = sorted(
            like_boo_info['boos'].most_common(),
            key=operator.itemgetter(1, 0),  # sort by count, then by user_id
            reverse=True,
        )[:10]
        for rank, (uid, count) in enumerate(top_10_boos):
            result[f'boo {rank+1}'] = {'user_id': uid, 'count': count}

        like_boo_file = f'push_{start_date}_{end_date}.json'
        with open(like_boo_file, 'w', encoding='utf-8') as file:
            json.dump(result, file, ensure_ascii=False)

    def add_like_boo_info(info1: dict, info2: dict):
        return {
            'all_like': info1['all_like'] + info2['all_like'],
            'all_boo': info1['all_boo'] + info2['all_boo'],
            'likes': info1['likes'] + info2['likes'],
            'boos': info1['boos'] + info2['boos'],
        }

    def sync_crawl(urls):
        results = [crawl_one_page(url) for url in tqdm(urls)]

        like_boo_info = reduce(add_like_boo_info, results, {
            'all_like': 0,
            'all_boo': 0,
            'likes': Counter(),
            'boos': Counter(),
        })
        dump_like_boo_info(like_boo_info)

    async def async_crawl(urls):
        results = await tqdm_asyncio.gather(
            *[asyncio.to_thread(crawl_one_page, url) for url in urls])

        like_boo_info = reduce(add_like_boo_info, results, {
            'all_like': 0,
            'all_boo': 0,
            'likes': Counter(),
            'boos': Counter(),
        })
        dump_like_boo_info(like_boo_info)

    urls = [article_info['url'] for article_info in article_infos]
    if use_async:
        asyncio.run(async_crawl(urls))
    else:
        sync_crawl(urls)


@cli.command(short_help='Find popular articles and its image urls')
@click.argument('start-date', type=str)
@click.argument('end-date', type=str)
@click.option('--use-async/--no-async', default=True, help='Use async to crawl')
def popular(start_date: str, end_date: str, use_async: bool):
    """Find popular articles and its image urls from START_DATE to END_DATE.

    \b
    Args:
        start_date (str): Start date.
        end_date (str): End date.
    """
    article_infos = load_article_infos(ALL_POPULAR_FILE, start_date, end_date)

    session = requests.Session()

    def is_image_url(url):
        prefix = ('http://', 'https://')
        suffix = ('jpg', 'jpeg', 'png', 'gif')
        return url.startswith(prefix) and url.endswith(suffix)

    def crawl_one_page(url: str):
        response = session.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'lxml')
        image_elements = soup.select('div#main-content a')

        image_urls = [
            i.get('href') for i in image_elements if is_image_url(i.get('href'))
        ]
        return image_urls

    def dump_image_urls(num_articles: int, image_urls: list):
        result = {
            'number_of_popular_articles': num_articles,
            'image_urls': image_urls,
        }
        image_urls_file = f'popular_{start_date}_{end_date}.json'
        with open(image_urls_file, 'w', encoding='utf-8') as file:
            json.dump(result, file, ensure_ascii=False)

    def sync_crawl(urls):
        results = [crawl_one_page(url) for url in tqdm(urls)]
        image_urls = reduce(operator.add, results, [])
        dump_image_urls(len(urls), image_urls)

    async def async_crawl(urls):
        results = await tqdm_asyncio.gather(
            *[asyncio.to_thread(crawl_one_page, url) for url in urls])
        image_urls = reduce(operator.add, results, [])
        dump_image_urls(len(urls), image_urls)

    urls = [article_info['url'] for article_info in article_infos]
    if use_async:
        asyncio.run(async_crawl(urls))
    else:
        sync_crawl(urls)


@cli.command(short_help='Find the images of the article contains the keyword')
@click.argument('kw', type=str)
@click.argument('start-date', type=str)
@click.argument('end-date', type=str)
@click.option('--use-async/--no-async', default=True, help='Use async to crawl')
def keyword(kw: str, start_date: str, end_date: str, use_async: bool):
    """Find the images of the article contains the KEYWORD from START_DATE to
    END_DATE.

    \b
    Args:
        kw (str): Keyword.
        start_date (str): Start date.
        end_date (str): End date.
    """
    article_infos = load_article_infos(ALL_ARTICLE_FILE, start_date, end_date)

    session = requests.Session()

    def contains_keyword(dom: BeautifulSoup):
        ele_list = dom.select('div#main-content span.f2')
        if not ele_list:
            return False
        if not any('發信站' in i.text for i in ele_list):
            return False

        contents = list(dom.select_one('div#main-content').children)
        end_idx = contents.index(ele_list[0])
        text = [i.text for i in contents[0:end_idx]]

        return any(kw in t for t in text)

    def is_image_url(url):
        prefix = ('http://', 'https://')
        suffix = ('jpg', 'jpeg', 'png', 'gif')
        return url.startswith(prefix) and url.endswith(suffix)

    def crawl_one_page(url: str):
        response = session.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'lxml')

        if not contains_keyword(soup):
            return []

        image_elements = soup.select('div#main-content a')
        image_urls = [
            i.get('href') for i in image_elements if is_image_url(i.get('href'))
        ]
        return image_urls

    def dump_image_urls(image_urls: list):
        result = {
            'image_urls': image_urls,
        }
        image_urls_file = f'keyword_{kw}_{start_date}_{end_date}.json'
        with open(image_urls_file, 'w', encoding='utf-8') as file:
            json.dump(result, file, ensure_ascii=False)

    def sync_crawl(urls):
        results = [crawl_one_page(url) for url in tqdm(urls)]
        image_urls = reduce(operator.add, results, [])
        dump_image_urls(image_urls)

    async def async_crawl(urls):
        results = await tqdm_asyncio.gather(
            *[asyncio.to_thread(crawl_one_page, url) for url in urls])
        image_urls = reduce(operator.add, results, [])
        dump_image_urls(image_urls)

    urls = [article_info['url'] for article_info in article_infos]
    if use_async:
        asyncio.run(async_crawl(urls))
    else:
        sync_crawl(urls)


if __name__ == '__main__':
    cli()
