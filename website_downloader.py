import asyncio
import argparse
from pyppeteer import launch


async def intercept(request):
    if any(request.resourceType == _ for _ in ("stylesheet", "image", "font")):
        await request.abort()
    else:
        await request.continue_()


async def download_pdf(page, url, output_file):
    # Launch a browser

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

    for i, url in enumerate(urls, start=1):
        output_file = f"{output_directory}/output_{i}.pdf"
        print(f"Downloading PDF for {url} to {output_file}")
        await download_pdf(page, url, output_file)

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
