import os
from typing import Any

import aiohttp
import requests
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


class Query(BaseModel):
    query: str = Field(description='需要查询的问题')

class SearchQueryByBoCha(BaseTool):
    name: str = 'SearchQueryByBoCha'
    description: str = '需要获取实时信息的时候，调用此工具'
    args_schema: type[Query] = Query
    return_direct:bool = True

    bocha_api_key: str = os.getenv("BOCHA_API_KEY")
    bocha_base_url: str = os.getenv("BOCHA_BASE_URL")

    @classmethod
    def _bocha_config_judge(cls) -> bool:
        return cls.bocha_api_key is not None and cls.bocha_base_url is not None

    @classmethod
    def get_session_kwargs(cls, query: str) -> dict:
        return {
            'url': cls.bocha_base_url,
            'headers': {
                'Content-Type': 'application/json; charset=utf-8',
                'Authorization': f'Bearer {cls.bocha_api_key}'
            },
            'json': {
                'query': query,
                'summary': True,
                'count': 9
            }
        }

    @staticmethod
    def response_analysis(res: Any) -> str:
        if str(res.get('code')) != '200':
            return f'请求错误：{res.get("message")}'

        webpages = res.get('data', {}).get('webpages', {}).get('values', [])
        if webpages is None:
            return '没有相关内容'

        formatted_webpages = []
        for index, webpage in enumerate(webpages, start=1):
            formatted_webpage = '\n'.join([
                f'编号：{index}',
                f'标题：{webpage.get('name')}',
                f'网址：{webpage.get("url")}',
                f'内容简述：{webpage.get('snippet')}',
                f'内容摘要：{webpage.get('summary')}',
            ])
            formatted_webpages.append(formatted_webpage)
        return '\n\n'.join(formatted_webpages)


    def _run(self, query: str) -> str:
        if not self._bocha_config_judge():
            raise ValueError('错误的博查配置')

        session_kwargs = self.get_session_kwargs(query)
        session_kwargs['timeout'] = 20

        resp = requests.post(**session_kwargs)
        resp.raise_for_status()
        res = resp.json()

        return self.response_analysis(res)

    async def _arun(self, query: str) -> str:
        if not self._bocha_config_judge():
            raise ValueError('错误的博查配置')

        session_kwargs = self.get_session_kwargs(query)
        session_kwargs['timeout'] = aiohttp.ClientTimeout(20)

        async with aiohttp.ClientSession() as session:
            async with session.get(**session_kwargs) as resp:
                resp.raise_for_status()
                res = await resp.json()

        return self.response_analysis(res)

