from pydantic_settings import BaseSettings



class Settings(BaseSettings):
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = '1234'
    POSTGRES_HOST: str = "localhost"
    POSTGRES_DB: str = "nzv_cmd"
    SPIDER_WAIT_TIMEOUT_SEC: int = 60 * 60 * 1  # каждый час
    SPIDER_PERIOD_DAYS: int = 10

    @property
    def DB_SQLALCHEMY_DATABASE_URL(self):
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}/{self.POSTGRES_DB}"
        )


APP_SETTINGS = Settings()
