import sqlite3
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

from career_copilot.models.job_offer import JobOffer


class JobRepository:

    def __init__(self, db_path: str = "career_copilot/data/jobs.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()

        self._create_table()
    
    def save(self, job: JobOffer):
        """Saves one job offer."""
        self.cursor.execute("""
            INSERT OR IGNORE INTO jobs (
                id,
                title,
                company,
                city,
                country,
                contract,
                publication_date,
                description,
                salary,
                url,
                source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job.id,
            job.title,
            job.company,
            job.city,
            job.country,
            job.contract,
            job.publication_date,
            job.description,
            job.salary,
            job.url,
            job.source,
        ))

        self.connection.commit()
        logger.info(f"Job saved: {job.title} at {job.company}")


    def save_all(self, jobs: list[JobOffer]) -> int:
        """Saves a list of job offers."""
        saved = 0

        for job in jobs:
            if not self.exists(job.url):
                self.save(job)
                saved += 1

        logger.info(f"Saved {saved} new job offers.")
        return saved

    def get_all(self) -> list[JobOffer]:
        """Return all job offers."""
        rows = self.cursor.execute("""
            SELECT * FROM jobs 
        """).fetchall()

        logger.info(f"Retrieved {len(rows)} job offers from the database.")
        return [self._row_to_job_offer(row) for row in rows]

    def get_by_id(self, job_id: str) -> JobOffer | None:
        """Return a job offer by its identifier."""
        row = self.cursor.execute("""
            SELECT * FROM jobs 
            WHERE id = ?
        """, (job_id,)).fetchone()

        if row is None :
            return None
        
        logger.info(f"Retrieved job offer with ID {job_id} from the database.")
        return self._row_to_job_offer(row)

    def exists(self, url: str) -> bool:
        """Check whether a job offer already exists using its URL."""
        row = self.cursor.execute("""
            SELECT * FROM jobs 
            WHERE url = ?
        """, (url,)).fetchone()

        logger.info(f"Job offer with URL {url} exists: {bool(row)}")
        return bool(row)

    def delete(self, job_id: str) -> None:
        """Delete a job offer by its identifier."""
        self.cursor.execute("""
            DELETE FROM jobs 
            WHERE id = ?
        """, (job_id,))

        logger.info(f"Deleted job offer with ID {job_id} from the database.")
        self.connection.commit()

    def count(self) -> int:
        """Return the total number of stored job offers."""
        count = self.cursor.execute("""
            SELECT COUNT(*) FROM jobs 
        """).fetchone()

        logger.info(f"Total number of job offers in the database: {count[0]}")
        return count[0]

    def search(self, keyword: str) -> list[JobOffer]:
        """Search job offers containing the given keyword."""
        rows = self.cursor.execute("""
            SELECT * FROM jobs 
            WHERE description LIKE ?
        """, (f"%{keyword}%", ))

        logger.info(f"Found {len(rows)} job offers containing the keyword '{keyword}'.")
        return [self._row_to_job_offer(row) for row in rows]
    
    def close(self) -> None:
        """Close the database connection."""
        self.connection.close()
        logger.info("Database connection closed.")

    def _create_table(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                title TEXT,
                company TEXT,
                city TEXT,
                country TEXT,
                contract TEXT,
                publication_date TEXT,
                description TEXT,
                salary TEXT,
                url TEXT UNIQUE,
                source TEXT
            )
        """)

        self.connection.commit()
        logger.info("Jobs table created or already exists.")

    def _row_to_job_offer(self, row) -> JobOffer:
        return JobOffer(
            id=row[0],
            title=row[1],
            company=row[2],
            city=row[3],
            country=row[4],
            contract=row[5],
            publication_date=row[6],
            description=row[7],
            salary=row[8],
            url=row[9],
            source=row[10],
        )