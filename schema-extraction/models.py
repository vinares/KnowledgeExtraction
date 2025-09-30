from pydantic import BaseModel
from typing import List, Dict
from enum import Enum
import dspy
import json
from pathlib import Path
import os

# Resolve the project root (the parent of "project")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEV_SET = os.path.join(PROJECT_ROOT, "data", "dev.json")
TRAIN_SET = os.path.join(PROJECT_ROOT, "data", "train.json")
TEST_SET = os.path.join(PROJECT_ROOT, "data", "test.json")

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
    Extract relation triplets from dialogue text between multiple speakers.
    
    CRITICAL RULES:
    1. NEVER use 'Speaker N' as entities in relations - always resolve to real named entities
    2. Extract relations ONLY between actual named entities mentioned in the text
    3. If a speaker refers to themselves by name, use that name instead of speaker label
    4. Focus on explicit relationships mentioned or strongly implied in the dialogue
    5. Only extract relations where both entities are properly identified (not generic terms)
    6. Use EXACT relation types from the provided list - no variations or abbreviations
    
    Entity Resolution Guidelines:
    - 'Speaker 1' → Use actual name if revealed (e.g., 'Joey' if Speaker 1 is Joey)
    - 'Speaker 2' → Use actual name if revealed (e.g., 'Phoebe' if Speaker 2 is Phoebe)  
    - Generic terms like 'man', 'woman', 'agent' should NOT be used as entities
    - Only use specific named persons, organizations, or clear string entities
    
    Relation Quality Rules:
    - Avoid trivial or obvious relations
    - Focus on relationships that convey meaningful information
    - Ensure relation types match the actual interaction described
    - Use EXACT relation type strings as provided in relation_types
    """
    episode_text: str = dspy.InputField(desc="Multi-speaker dialogue episode")
    entity_types: List[str] = dspy.InputField(
        desc="Allowed entity types: per (person), org (organization), string (other)",
        default=ALL_ENTITY_TYPES
    )
    relation_types: List[str] = dspy.InputField(
        desc=(
            "Allowed relation types in format 'entity:relation'. "
            "MUST USE EXACTLY THESE STRINGS: " + ", ".join(ALL_RELATION_TYPES)
        ),
        default=ALL_RELATION_TYPES
    )
    relation_triplets: str = dspy.OutputField(
        desc=(
            "JSON string representing a list of relation triplets with real entity names. "
            "CRITICAL: NEVER use 'Speaker N' as entities. Always resolve to actual named entities. "
            "If speaker identities are unclear from context, omit the relation. "
            "MUST USE EXACT relation types from the provided list. "
            "Example output format: "
            '[{"x": "Joey Tribbiani", "y": "Phoebe Buffay", "r": "per:friends", "x_type": "per", "y_type": "per"}, '
            '{"x": "Joey Tribbiani", "y": "Estelle", "r": "per:employee_of", "x_type": "per", "y_type": "per"}]'
            "\n\nBAD examples to avoid: "
            "- Relations with 'Speaker 1', 'Speaker 2', etc."
            "- Relations with generic terms like 'man', 'woman', 'agent' as entities"
            "- Using relation types not in the provided list"
            "- Trivial or unsubstantiated relations"
        )
    )

class OptimizedRelationExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(DialogRERelation)
    
    def forward(self, episode_text):
        try:
            result = self.predictor(
                episode_text=episode_text,
                entity_types=ALL_ENTITY_TYPES,
                relation_types=ALL_RELATION_TYPES,
            )
            
            # Parse the JSON string output
            triplets_data = json.loads(result.relation_triplets)
            
            # Convert to RelationTriplet objects with validation
            validated_triplets = []
            for item in triplets_data:
                try:
                    # Validate and convert relation type
                    if item["r"] not in ALL_RELATION_TYPES:
                        # Try to find the closest match
                        corrected_relation = self._correct_relation_type(item["r"])
                        if corrected_relation:
                            item["r"] = corrected_relation
                        else:
                            print(f"Skipping invalid relation type: {item['r']}")
                            continue
                    
                    # Validate entity types
                    if item["x_type"] not in ALL_ENTITY_TYPES:
                        item["x_type"] = self._correct_entity_type(item["x_type"])
                    if item["y_type"] not in ALL_ENTITY_TYPES:
                        item["y_type"] = self._correct_entity_type(item["y_type"])
                    
                    # Skip speaker entities and generic terms
                    if self._is_speaker_entity(item["x"]) or self._is_speaker_entity(item["y"]):
                        continue
                    if self._is_generic_entity(item["x"]) or self._is_generic_entity(item["y"]):
                        continue
                    
                    # Create the validated triplet
                    triplet = RelationTriplet(
                        x=item["x"],
                        y=item["y"], 
                        r=RelationType(item["r"]),
                        x_type=EntityType(item["x_type"]),
                        y_type=EntityType(item["y_type"])
                    )
                    validated_triplets.append(triplet)
                    
                except (KeyError, ValueError) as e:
                    print(f"Skipping invalid triplet {item}: {e}")
                    continue
                    
            return validated_triplets
            
        except (json.JSONDecodeError, TypeError, AttributeError) as e:
            print(f"Error parsing response: {e}")
            print(f"Raw response: {getattr(result, 'relation_triplets', 'No response')}")
            return []
    
    def _correct_relation_type(self, relation_str):
        """Try to correct common relation type mistakes"""
        corrections = {
            "per:friend": "per:friends",
            "per:employee": "per:employee_of", 
            "per:student": "per:student_of",
            "per:member": "per:member_of",
            "per:title_of": "per:title",
            "per:alternate_name": "per:alternate_names",
        }
        return corrections.get(relation_str)
    
    def _correct_entity_type(self, entity_type_str):
        """Correct common entity type mistakes"""
        if entity_type_str.lower() in ["person", "per"]:
            return "per"
        elif entity_type_str.lower() in ["organization", "org"]:
            return "org" 
        elif entity_type_str.lower() in ["string", "str", "other"]:
            return "string"
        return "string"  # default fallback
    
    def _is_speaker_entity(self, entity):
        """Check if entity is a speaker label"""
        return isinstance(entity, str) and "Speaker" in entity and any(char.isdigit() for char in entity)
    
    def _is_generic_entity(self, entity):
        """Check if entity is too generic"""
        if not isinstance(entity, str):
            return True
        generic_terms = {
            'man', 'woman', 'person', 'agent', 'director', 'office', 
            'problem', 'someone', 'anybody', 'everyone', 'people'
        }
        return entity.lower() in generic_terms