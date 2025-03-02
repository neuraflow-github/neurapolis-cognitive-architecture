import logging

import bugsnag
from langchain_core.messages import HumanMessage
from neurapolis_cognitive_architecture.models import MyHumanMessage, State
from neurapolis_common import File, Neo4jDbSessionBuilder, get_last_message_of_type


class MentionedFilesRetrieverNode:
    async def mentioned_files_retriever(self, state: State) -> dict:
        logging.info(f"{self.__class__.__name__}: Started")

        try:
            last_human_message: MyHumanMessage = get_last_message_of_type(
                state["messages"], MyHumanMessage
            )

            # TODO Remove
            print(last_human_message.mentioned_file_ids)
            if len(last_human_message.mentioned_file_ids) == 0:
                content = "Der Nutzer hat sich auf keine Dateien bezogen."
            else:
                print("Hi 1")
                async with Neo4jDbSessionBuilder().build() as neo4j_db_session:
                    neo4j_db_query = """
                    MATCH (file_node:File)
                    WHERE file_node.id IN $file_ids
                    RETURN file_node
                    """
                    neo4j_db_results = await neo4j_db_session.run(
                        neo4j_db_query, file_ids=last_human_message.mentioned_file_ids
                    )

                    files: list[File] = []
                    async for x_neo4j_db_result in neo4j_db_results:
                        file = File.create_from_neo4j_db_node(
                            x_neo4j_db_result["file_node"]
                        )
                        files.append(file)

                print("Hi 2")
                print(files)
                inner_xml = File.format_multiple_to_inner_llm_xml(files)
                xml = f"<{File.get_llm_xml_tag_name_prefix()}>\n{inner_xml}\n</{File.get_llm_xml_tag_name_prefix()}>"
                found_files_content = (
                    f"Der Nutzer hat sich auf die folgenden Dateien bezogen:\n\n{xml}"
                )

                found_file_ids: list[str] = []
                for x_file in files:
                    found_file_ids.append(x_file.id)

                not_found_file_ids: list[str] = []
                for x_file_id in last_human_message.mentioned_file_ids:
                    if x_file_id in found_file_ids:
                        continue

                    not_found_file_ids.append(x_file_id)

                content = found_files_content

                if len(not_found_file_ids) > 0:
                    not_found_file_ids_content = f"Der Nutzer hat sich außerdem auf die Dateien mit den folgenden IDs bezogen, welche aber nicht in der Datenbank sind: {', '.join(not_found_file_ids)}"
                    content += f"\n\n\n{not_found_file_ids_content}"

                print("Hi 3")
            print(content)

            message = HumanMessage(content=content)
        except Exception as exception:
            logging.error(f"{self.__class__.__name__}: Failed", exc_info=True)
            bugsnag.notify(exception)

            raise exception

        logging.info(f"{self.__class__.__name__}: Finished")

        return {"messages": [message]}
