from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from loguru import logger

from app.api import api_router
from app.config import settings, setup_app_logging

# setup logging as early as possible
setup_app_logging(config=settings)


app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

root_router = APIRouter()


@root_router.get("/")
def index(request: Request) -> Any:
    """Home end point"""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to the En-Ar Translator API</h1>"
        """<div> You've reached the En-Ar Translator API:
        <a href='/docs'>Proceed</a> </div>"""
        "<div>"
        "<div>"
        "</div>"
        "Using the api interface:"
        "<ul>"
        """<li> Open the 'proceed' link in a new tab and keep this one to follow the
        rest of the instructions. </li>"""
        "<li> Expand the 'POST' request section</li>"
        "<li> Find the 'try it out' button. </li>"
        """<li> Edit the 'Request body' input box to your liking then
        hit 'execute'. </li>"""
        "<li> Scroll down further (ignore the cURL request template following it) </li>"
        "<li> Check the translation under the reponse body .</li>"
        """<li> Note 1: status code 200 means the post request was valid and should
        yield a translation within the response body. </li>"""
        """<li> Note 2: This tanslator was trained on 10,000 En-Ar sentence pairs
        and so it not the most sophisticated. It an handle short sentences.
        This translator is being further trained on more En-Ar sentence pairs as they
        are collected.</li>"""
        "</ul>"
        "</div>"
        "</body>"
        "</html>"
    )

    return HTMLResponse(content=body)  # return a webpage response


# route to health and predict end points
app.include_router(api_router, prefix=settings.API_V1_STR)
# route to home end point
app.include_router(root_router)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        # admits various origins (protocols/ports/etc.) if front end and
        # backend attempt to communicate
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        # allows any request method (post/get)
        allow_methods=["*"],
        # allows request with any header like cookies/User-Agent; can disable
        # not browser user-agents for example
        allow_headers=["*"],
    )


if __name__ == "__main__":
    # Use this for debugging purposes only
    logger.warning("Running in development mode.")
    import uvicorn

    # log level does not follow production api log level since .run is
    # for debugging in development; production api captures INFO level
    # uvicorn logs using Loguru
    uvicorn.run(app, host="localhost", port=8001, log_level="debug")
