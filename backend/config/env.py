from pydantic_settings import BaseSettings, SettingsConfigDict


class App_Config(BaseSettings):

    # Frontend
    NEXT_PUBLIC_FRONTEND_URL: str

    # Backend
    NEXT_PUBLIC_BACKEND_URL: str
    
    # Google Places API
    GOOGLE_PLACES_API_KEY: str

    # Config for loading environment variables
    model_config = SettingsConfigDict(env_file="../.env", env_file_encoding="utf-8")


def get_app_configs() -> App_Config:
    return App_Config()