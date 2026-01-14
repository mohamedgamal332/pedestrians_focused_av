from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from FlagEmbedding import BGEM3FlagModel
import json

class ExperienceMemory:
    def __init__(self, host="localhost", port="19530"):
        # 1. Initialize Pretrained BGE-M3
        self.encoder = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        
        # 2. Connect to Milvus
        connections.connect("default", host=host, port=port)
        self.collection = self._setup_collection()

    def _setup_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024), # BGE-M3 Dim
            FieldSchema(name="raw_data", dtype=DataType.VARCHAR, max_length=65535)
        ]
        schema = CollectionSchema(fields, "Autonomous Driving Experience Logs")
        return Collection("driving_logs", schema)

    def commit(self, reasoning_trace, reflex_action, safety_score):
        """Converts the scene into a searchable vector."""
        # Create a text representation of the moment for BGE-M3
        scene_desc = f"Reasoning: {reasoning_trace} | Action: {reflex_action} | Safety: {safety_score}"
        
        vector = self.encoder.encode([scene_desc])['dense_vecs'][0]
        
        # Store metadata as JSON
        metadata = json.dumps({
            "trace": reasoning_trace,
            "action": reflex_action,
            "score": safety_score
        })
        
        self.collection.insert([ [vector], [metadata] ])
        self.collection.flush()