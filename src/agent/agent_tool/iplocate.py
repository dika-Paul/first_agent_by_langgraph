import os
from typing import Any

import aiohttp
import requests
from pydantic import Field, BaseModel
from langchain_core.tools import BaseTool


class IPModel(BaseModel):
    ip_address: str = Field(description="IP地址")

class IPLocateByGaoDe(BaseTool):
    name: str = "IPLocateByGaoDe"
    description: str = "需要根据IP地址获得定位时，调用此工具"
    args_schema: type[IPModel] = IPModel
    return_direct: bool = True
    gaode_api_key: str = os.getenv("GAODE_API_KEY")
    gaode_base_url: str = os.getenv("GAODE_BASE_URL")

    @classmethod
    def _gaode_config_judge(cls) -> bool:
        return cls.gaode_api_key is not None and cls.gaode_base_url is not None

    @classmethod
    def get_session_kwargs(cls, ip_address:str) -> dict[str, Any]:
        return {
            "url": f'{cls.gaode_base_url}ip={ip_address}&key={cls.gaode_api_key}',
            "headers": {'Content-Type': 'application/json; charset=utf-8'},
        }

    @staticmethod
    def response_analysis(res: Any) -> str:
        if res.get('status') == 0:
            return f'查询失败：{res.get('info', '未知错误')}'

        province = res.get('province')
        city = res.get('city')

        if province is None and city is None:
            return '非法或中国外IP'

        return f'{province}, {city}'


    def _run(self, ip_address: str) -> str:
        if not self._gaode_config_judge():
            raise ValueError('错误的高德配置')

        session_kwargs = self.get_session_kwargs(ip_address)
        session_kwargs['timeout'] = 20

        resp = requests.get(**session_kwargs)
        resp.raise_for_status()
        res = resp.json()

        return self.response_analysis(res)


    async def _arun(self, ip_address: str) -> str:
        if not self._gaode_config_judge():
            raise ValueError('错误的高德配置')

        session_kwargs = self.get_session_kwargs(ip_address)
        session_kwargs['timeout'] = aiohttp.ClientTimeout(20)

        async with aiohttp.ClientSession() as session:
            async with session.get(**session_kwargs) as resp:
                resp.raise_for_status()
                res = await resp.json()

        return self.response_analysis(res)

