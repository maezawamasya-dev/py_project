"""
Клиент для NLP микросервиса
"""

import requests
import json
import time
import sys
from typing import Dict, Any, Optional

class NLPClient:
    """Клиент для взаимодействия с NLP микросервисом"""
    
    def init(self, base_url: str = "http://localhost:8000"):
        """
        Инициализация клиента
        
        Args:
            base_url: Базовый URL сервера
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "NLP-Microservice-Client/1.0"
        })
    
    def check_connection(self) -> bool:
        """
        Проверка подключения к серверу
        
        Returns:
            bool: True если сервер доступен
        """
        try:
            response = self.session.get(f"{self.base_url}/", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def get_server_info(self) -> Dict[str, Any]:
        """
        Получение информации о сервере
        """
        try:
            response = self.session.get(f"{self.base_url}/")
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Connection failed: {str(e)}"}
    
    def get_tfidf(self) -> Dict[str, Any]:
        """
        Получение TF-IDF матрицы
        """
        try:
            response = self.session.post(f"{self.base_url}/tf-idf")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def bag_of_words(self, text: str) -> Dict[str, Any]:
        """
        Преобразование текста в Bag-of-Words
        
        Args:
            text: Текст для обработки
        """
        try:
            response = self.session.get(
                f"{self.base_url}/bag-of-words",
                params={"text": text}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def lsa_analysis(self, n_components: int = 2) -> Dict[str, Any]:
        """
        Латентный семантический анализ
        
        Args:
            n_components: Количество компонент
        """
        try:
            response = self.session.post(
                f"{self.base_url}/lsa",
                params={"n_components": n_components}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def tokenize(self, text: str) -> Dict[str, Any]:
        """
        Токенизация текста
        
        Args:
            text: Текст для токенизации
        """
        try:
            response = self.session.post(
                f"{self.base_url}/text_nltk/tokenize",
                data={"text": text}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def stem(self, text: str) -> Dict[str, Any]:
        """
        Стемминг текста
        
        Args:
            text: Текст для стемминга
        """
        try:
            response = self.session.post(
                f"{self.base_url}/text_nltk/stem",
                data={"text": text}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def lemmatize(self, text: str) -> Dict[str, Any]:
        """
        Лемматизация текста
