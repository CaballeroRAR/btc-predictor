from google.cloud import firestore
import os
from src.repositories.base import BaseRepository
import cloud_config as cloud_config

class FirestoreRepository(BaseRepository):
    """
    Concrete implementation of the Repository Pattern for Google Cloud Firestore.
    """
    def __init__(self):
        super().__init__("repositories.firestore")
        db_id = os.getenv("FIRESTORE_DATABASE", cloud_config.FIRESTORE_DATABASE)
        self.db = firestore.Client(project=cloud_config.PROJECT_ID, database=db_id)
        self.logger.info(f"Firestore Client connected to project: {cloud_config.PROJECT_ID}, db: {db_id}")

    def save(self, data: dict, collection: str, document_id: str = None):
        """Standardized save operation with industrial logging and local fallback."""
        try:
            self.logger.info(f"Saving document to {collection}/{document_id or 'auto-gen'}")
            col_ref = self.db.collection(collection)
            if document_id:
                doc_ref = col_ref.document(str(document_id))
            else:
                doc_ref = col_ref.document()
            
            data['updated_at'] = firestore.SERVER_TIMESTAMP
            doc_ref.set(data, merge=True)
            self.logger.debug(f"Successfully committed data to {collection}")
            return doc_ref.id
        except Exception as e:
            self.logger.error(f"Firestore save failed: {str(e)}. Triggering Resilience Layer.")
            return self._save_local(data, collection, document_id)

    def get(self, collection: str, document_id: str):
        """Retrieve a single document with local fallback."""
        try:
            self.logger.info(f"Fetching document {collection}/{document_id}")
            doc = self.db.collection(collection).document(document_id).get()
            if doc.exists:
                self.logger.debug(f"Document found: {document_id}")
                return doc.to_dict()
            self.logger.warning(f"Document not found in cloud: {document_id}")
        except Exception as e:
            self.logger.error(f"Firestore fetch failed: {str(e)}. Triggering Resilience Layer.")
            
        return self._get_local(collection, document_id)

    def query(self, collection: str, filters: list = None, order_by: str = None, limit: int = None):
        """Advanced query implementation."""
        try:
            self.logger.info(f"Querying collection {collection} with {len(filters or [])} filters")
            ref = self.db.collection(collection)
            
            if filters:
                for f in filters:
                    ref = ref.where(f[0], f[1], f[2])
            
            if order_by:
                ref = ref.order_by(order_by)
            
            if limit:
                ref = ref.limit(limit)
                
            results = [doc.to_dict() for doc in ref.stream()]
            self.logger.info(f"Query returned {len(results)} results")
            return results
        except Exception as e:
            self.logger.error(f"Query failed: {str(e)}")
            return []

    def delete(self, collection: str, document_id: str):
        """Delete a document."""
        try:
            self.logger.warning(f"DELETING document {collection}/{document_id}")
            self.db.collection(collection).document(str(document_id)).delete()
            self.logger.info(f"Document {document_id} deleted successfully")
            return True
        except Exception as e:
            self.logger.error(f"Deletion failed: {str(e)}")
            return False

    def get_latest_snapshot(self):
        """Retrieve the most recent forecasting snapshot from the snapshots collection."""
        self.logger.info("Querying latest forecasting snapshot")
        try:
            docs = self.db.collection('snapshots').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1).stream()
            for doc in docs:
                data = doc.to_dict()
                data['document_id'] = doc.id
                return data
        except Exception as e:
            self.logger.error(f"Failed to retrieve latest snapshot: {str(e)}")
        return None

    def save_system_snapshot(self, results_data: dict):
        """Persist a complete forecasting result as a system snapshot."""
        self.logger.info("Persisting new system forecasting snapshot")
        # Ensure timestamp is set for ordering
        results_data['timestamp'] = firestore.SERVER_TIMESTAMP
        return self.save(results_data, 'snapshots')
