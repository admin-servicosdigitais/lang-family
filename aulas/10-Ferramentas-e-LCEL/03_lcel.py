"""Passo 3: compor etapas com LCEL, usando o operador ``|``."""

from langchain_core.language_models import FakeListChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

PROMPT = ChatPromptTemplate.from_messages(
    [("system", "Resuma a mensagem do cliente em poucas palavras."), ("human", "{message}")]
)


def main() -> None:
    model = FakeListChatModel(responses=["pedido atrasado", "cobrança em dobro"])
    # Cada etapa recebe a saída da anterior.
    chain = PROMPT | model | StrOutputParser() | RunnableLambda(str.upper)

    print(
        f"prompt montado: {PROMPT.invoke({'message': 'cadê meu pedido?'}).to_messages()[-1].text}"
    )
    print(f"invoke: {chain.invoke({'message': 'cadê meu pedido?'})}")
    model.i = 0
    # A mesma cadeia ganha batch (e stream e as versões assíncronas) sem código extra.
    inputs = [{"message": "cadê meu pedido?"}, {"message": "cobraram duas vezes"}]
    print(f"batch : {chain.batch(inputs, {'max_concurrency': 1})}")


if __name__ == "__main__":
    main()
