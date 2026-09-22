from playwright.sync_api import sync_playwright
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)

from career_copilot.models.job_offer import JobOffer

class LinkedInCollector:
    BASE_URL = "https://www.linkedin.com/jobs/search"

    def collect(
        self,
        keywords: str,
        location: str,
        limit: int = 20
    ) -> list[JobOffer]:

        jobs: list[JobOffer] = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()

            #fill the space in the url
            search_url = (
                f"{self.BASE_URL}"
                f"?keywords={keywords.replace(' ', '%20')}"
                f"&location={location.replace(' ', '%20')}"
            )

            page.goto(search_url)
            page.wait_for_timeout(5000)

            #search of the list of offers with <li class="base-card">
            cards = page.locator(".base-card").all()

            for card in cards[:limit]:
                try:
                    jobs.append(self._parse_job_details(card, page))
                except Exception as e:
                    print(e)
                    logger.warning(f"Failed to parse job details for card: {card}")
                    
        logger.info(f"Collected {len(jobs)} job offers from LinkedIn for keywords '{keywords}' in location '{location}'.")

        return jobs
    
    def _parse_job_details(self, card, page) -> JobOffer:
        """Extract the details of a job offer from a LinkedIn job card and its corresponding job page."""

        # Informations of the card

        title = card.locator("h3").inner_text().strip()

        company = card.locator("h4").inner_text().strip()

        location = card.locator(".job-search-card__location").inner_text().strip()

        city = location.split(",")[0]
        country = location.split(",")[-1].strip()

        url = card.locator("a").first.get_attribute("href")

        # Offer opening

        card.click()

        page.wait_for_load_state("networkidle")

        # Offer description

        description = None
        contract = None
        publication_date = None
        salary = None

        try:
            description = (
                page.locator(".show-more-less-html__markup")
                .inner_text()
                .strip()
            )
        except Exception:
            pass

        try:
            publication_date = (
                page.locator("time")
                .first
                .get_attribute("datetime")
            )
        except Exception:
            pass

        logger.debug(f"Job details extracted: {title} at {company} in {city}, {country}. URL: {url}")

        return JobOffer(
            id=str(uuid4()),
            title=title,
            company=company,
            city=city,
            country=country,
            contract=contract,
            publication_date=publication_date,
            description=description,
            salary=salary,
            url=url,
            source="linkedin",
        )


if __name__ == "__main__":
    
    collector = LinkedInCollector()

    jobs = collector.collect(
        keywords="Data Scientist",
        location="Lyon",
    )

    for job in jobs:
        print(job)