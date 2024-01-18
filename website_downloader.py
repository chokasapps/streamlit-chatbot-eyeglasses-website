import asyncio
import argparse
from pyppeteer import launch
import html2text
import aiofiles as aiof
import bleach
from bs4 import BeautifulSoup


async def intercept(request):
    if any(request.resourceType == _ for _ in ("stylesheet", "image", "font")):
        await request.abort()
    else:
        await request.continue_()


async def download_text(page, url, output_directory, i, text_maker):
    output_file = f"{output_directory}/output_{i}.txt"
    print(f"Downloading Text for {url} to {output_file}")
    options = {"timeout": 60000 * 5}  # 60000 * 10 is 5 minute

    # Navigate to the URL
    await page.goto(url, options=options)
    content = await page.content()
    soup = BeautifulSoup(content, "html.parser")
    for div in soup.find_all("div", {"class": "custom-header"}):
        div.decompose()
    main_html = soup.select("div.container-full.site-content")[0].prettify()
    page_text = text_maker.handle(main_html)
    page_text = bleach.clean(page_text)
    out = await aiof.open(output_file, "w", encoding="utf-8")
    await out.write(page_text)
    await out.flush()


async def download_pdf(page, url, output_directory, i):
    output_file = f"{output_directory}/output_{i}.pdf"
    print(f"Downloading PDF for {url} to {output_file}")

    options = {"timeout": 60000 * 5}  # 60000 * 10 is 5 minute

    # Navigate to the URL
    await page.goto(url, options=options)

    # Generate PDF from the page
    await page.pdf({"path": output_file, "format": "A4"})


async def process_urls(file_path, output_directory, limit=400):
    with open(file_path, "r") as file:
        urls = file.read().splitlines()

    # Limit the number of URLs to 400
    urls = urls[:limit]

    browser = await launch()

    # Create a new page
    page = await browser.newPage()

    # Disable images on the page
    await page.setRequestInterception(True)
    page.on("request", lambda req: asyncio.ensure_future(intercept(req)))

    text_maker = html2text.HTML2Text()
    text_maker.ignore_links = True
    text_maker.bypass_tables = False
    text_maker.ignore_images = True

    for i, url in enumerate(urls, start=1):
        await download_text(page, url, output_directory, i, text_maker)

    # Close the browser
    await browser.close()


def main():
    parser = argparse.ArgumentParser(description="Download PDFs from a list of URLs.")
    parser.add_argument("file_path", type=str, help="Path to the file containing URLs.")
    parser.add_argument(
        "output_directory", type=str, help="Directory to save downloaded PDFs."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=400,
        help="Limit the number of URLs to download (default: 400)",
    )

    args = parser.parse_args()

    # Run the asyncio event loop
    asyncio.get_event_loop().run_until_complete(
        process_urls(args.file_path, args.output_directory, args.limit)
    )


if __name__ == "__main__":
    main()
