import requests
import copy

class Wikidata:
    def __init__(self):
        """
        Initialize the Wikidata class.
        This class provides methods to interact with the Wikidata API, allowing
        retrieval of entities and their properties based on labels or IDs.
        """
        pass

    def get_id(self, label):
        """
        Retrieve the Wikidata ID of an entity using a label.
        
        Args:
            label (str): The label to search for in Wikidata.
        
        Returns:
            str: The Wikidata ID of the entity, or None if the entity is not found.
        """
        response = self._search_label(label)
        if not response:
            return

        search_results = response["search"]
        return search_results[0]["id"] if search_results else None

    def from_label(self, label):
        """
        Retrieve entity information from Wikidata using a label.
        
        Args:
            label (str): The label to search for in Wikidata.
        
        Returns:
            list: A list of triples representing the entity's properties and qualifiers.
        """
        response = self._search_label(label)
        if not response:
            return

        search_results = response["search"]
        qid = search_results[0]["id"]  # Get the first search result's ID

        return self.from_id(qid)

    def from_id(self, qid):
        """
        Retrieve entity information from Wikidata using an entity ID.
        
        Args:
            qid (str): The Wikidata ID of the entity.
        
        Returns:
            list: A list of triples representing the entity's properties and qualifiers.
        """
        response = self._get_entity(qid)

        if not response:
            return

        triples = []

        entity_info = response["entities"][qid]

        try:
            entity_label = entity_info["labels"]["en"]["value"]
        except KeyError:
            return  # Return None if the entity label is not found

        try:
            entity_description = entity_info["descriptions"]["en"]["value"]
            triples.append([[entity_label, "is", entity_description]])
        except KeyError:
            pass  # Continue if the description is not available

        entity_claims = entity_info["claims"]

        for property_id, statements in entity_claims.items():
            property_label = self._id_to_label(property_id)

            for statement in statements:
                mainsnak = statement["mainsnak"]
                if mainsnak["snaktype"] != "value":
                    continue  # Skip if the statement does not have a valid value

                mainsnak_value = self._get_snak_value(mainsnak)
                if not mainsnak_value:
                    continue  # Skip if the value cannot be retrieved

                triple = [[entity_label, property_label, mainsnak_value]]

                # Process qualifiers if available
                if "qualifiers" not in statement:
                    triples.append(triple)
                    continue

                qualifiers = statement["qualifiers"]

                for qualifier_property_id, qualifier_snaks in qualifiers.items():
                    qualifier_property_label = self._id_to_label(qualifier_property_id)

                    for qualifier_snak in qualifier_snaks:
                        qualifier_snak_value = self._get_snak_value(qualifier_snak)
                        if not qualifier_snak_value:
                            continue  # Skip if the qualifier value cannot be retrieved

                        qualifier = [qualifier_property_label, qualifier_snak_value]
                        triple.append(qualifier)

                triples.append(triple)

        return triples

    def verbalize_triples(self, triples):
        """
        Convert a list of triples into a human-readable string format.
        
        Args:
            triples (list): A list of triples, where each triple is a list of lists
                            containing entity label, relation, and value.
        
        Returns:
            list: A list of strings, each representing a verbalized triple.
        """
        verbalized_triples = copy.deepcopy(triples)

        for i in range(len(verbalized_triples)):
            triple = verbalized_triples[i]

            for j in range(len(triple)):
                triple[j] = ", ".join(triple[j])

            triple = "; ".join(triple)
            verbalized_triples[i] = f"({triple})"

        return verbalized_triples

    def _valid_response(self, response):
        """
        Check if the response from the API is valid (status code 200).
        
        Args:
            response (requests.Response): The response object from an API request.
        
        Returns:
            bool: True if the response is valid, False otherwise.
        """
        return response.status_code == 200

    def _search_label(self, label):
        """
        Search for a Wikidata entity using a label.
        
        Args:
            label (str): The label to search for in Wikidata.
        
        Returns:
            dict: The search results from Wikidata.
        """
        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": label,
        }

        response = requests.get(url, params=params)
        if self._valid_response(response):
            return response.json()

    def _get_entity(self, id):
        """
        Retrieve an entity's information from Wikidata using its ID.
        
        Args:
            id (str): The Wikidata ID of the entity.
        
        Returns:
            dict: The entity information in JSON format.
        """
        url = f"https://www.wikidata.org/wiki/Special:EntityData/{id}.json"

        response = requests.get(url)
        if self._valid_response(response):
            return response.json()

    def _id_to_label(self, id):
        """
        Convert a Wikidata property or entity ID to its corresponding label.
        
        Args:
            id (str): The Wikidata ID.
        
        Returns:
            str: The label corresponding to the ID, or None if not found.
        """
        response = self._get_entity(id)
        if response:
            try:
                return response["entities"][id]["labels"]["en"]["value"]
            except KeyError:
                return None

    def _get_snak_value(self, snak):
        """
        Extract the value from a Wikidata snak.
        
        Args:
            snak (dict): The snak object containing the value and datatype.
        
        Returns:
            str: The extracted value in a human-readable format, or None if the value could not be processed.
        """
        try:
            datatype = snak["datatype"]
            datavalue = snak["datavalue"]["value"]
        except KeyError:
            return None

        match datatype:
            case "wikibase-item":
                tail_id = datavalue["id"]
                snak_value = self._id_to_label(tail_id)

            case "time":
                snak_value = datavalue["time"]

            case "string":
                snak_value = datavalue

            case "quantity":
                amount = datavalue["amount"]
                unit = datavalue["unit"]

                if unit == "1":
                    snak_value = amount
                elif "wikidata.org/entity" in unit:
                    unit_id = unit.split("/")[-1]
                    unit_label = self._id_to_label(unit_id)
                    snak_value = amount + " " + unit_label

            case "globe-coordinate":
                latitude = datavalue["latitude"]
                longitude = datavalue["longitude"]
                snak_value = f"{latitude}, {longitude}"

            case "math":
                snak_value = datavalue

            case _:
                snak_value = None

        return snak_value
