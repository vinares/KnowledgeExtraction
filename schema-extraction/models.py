from pydantic import BaseModel
from typing import List, Dict
from enum import Enum
import dspy
import json


DEV_SET = "data/dev.json"
TRAIN_SET = "data/train.json"
TEST_SET = "data/test.json"

class EntityType(str, Enum):
    per = "per"
    org = "org"
    string = "string"

ALL_ENTITY_TYPES = [e.value for e in EntityType]

class RelationType(str, Enum):
    per_title = "per:title"
    per_alternate_names = "per:alternate_names"
    per_employee_of = "per:employee_of"
    per_student_of = "per:student_of"
    per_member_of = "per:member_of"
    per_origin = "per:origin"
    per_spouse = "per:spouse"
    per_children = "per:children"
    per_parents = "per:parents"
    per_siblings = "per:siblings"
    per_other_family = "per:other_family"
    per_friends = "per:friends"
    per_schools_attended = "per:schools_attended"
    per_places_lived = "per:places_lived"
    per_date_of_birth = "per:date_of_birth"
    per_age = "per:age"
    per_religion = "per:religion"
    per_political_affiliation = "per:political_affiliation"
    per_country_of_citizenship = "per:country_of_citizenship"
    per_state_of_residence = "per:state_of_residence"
    per_city_of_residence = "per:city_of_residence"
    per_language = "per:language"
    per_profession = "per:profession"
    per_death_date = "per:death_date"
    per_death_place = "per:death_place"
    org_top_members_employees = "org:top_members/employees"
    org_number_of_employees_members = "org:number_of_employees/members"
    org_member_of = "org:member_of"
    org_subsidiaries = "org:subsidiaries"
    org_parents = "org:parents"
    org_founded_by = "org:founded_by"
    org_place_of_headquarters = "org:place_of_headquarters"
    org_date_founded = "org:date_founded"
    org_political_religious_affiliation = "org:political/religious_affiliation"
    org_shareholders = "org:shareholders"

ALL_RELATION_TYPES = [r.value for r in RelationType]

class RelationTriplet(BaseModel):
    x: str
    y: str
    r: RelationType
    x_type: EntityType
    y_type: EntityType

class DialogRERelation(dspy.Signature):
    """
    Extract relation triplets from dialogue text.
    Focus on the input episode_text, don't refer to the internet or any external knowledge.
    """
    episode_text: str = dspy.InputField(desc="Dialogue episode between speakers")
    entity_types: List[str] = dspy.InputField(
        desc="Allowed entity types: per (person), org (organization), string (other)",
        default=ALL_ENTITY_TYPES
    )
    relation_types: List[str] = dspy.InputField(
        desc="Allowed relation types in format 'entity:relation'",
        default=ALL_RELATION_TYPES
    )
    relation_triplets: List[RelationTriplet] = dspy.OutputField(
        desc=(
            "List of relation triplets with x:string, y:string, r:relation_types, x_type:entity_types, y_type:entity_types fields. "
            "Example: "
            '[{"x": "Joey", "y": "Ross", "r": "per:friends", "x_type": "per", "y_type": "per"}, '
            '{"x": "Speaker 1", "y": "Joey", "r": "per:alternate_names", "x_type": "per", "y_type": "per"}]'
            "Deduce relations between real entities and drop 'Speaker N' entities."
        )
    )

class OptimizedRelationExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        # Examples are used during initialization to teach the pattern
        # but are NOT passed with every prediction call
        self.predictor = dspy.Predict(DialogRERelation)
    
    def forward(self, episode_text):
        # During prediction, we only pass the current input
        # The model has already learned from the examples during initialization
        result = self.predictor(
            episode_text=episode_text,
            entity_types=ALL_ENTITY_TYPES,
            relation_types=ALL_RELATION_TYPES,
        )
        
        try:
            # result.relation_triplets already contains RelationTriplet objects
            return result.relation_triplets
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error parsing response: {e}")
            return []