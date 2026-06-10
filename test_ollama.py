from openai import AsyncOpenAI
import asyncio
import time


async def main():
    client = AsyncOpenAI(
        api_key="f79ca5fc88bd42a3906cd50036ce7951.o9q5geJj5msh3gylXoxXVhpw",
        base_url="https://api.ollama.cloud/v1",
    )

    start = time.time()

    response = await client.chat.completions.create(
        model="gemma3:12b",
        messages=[{"role": "user", "content": "Reply with exactly 'Hi'"}],
    )   

    print(f"Response time: {time.time() - start:.2f}s")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    asyncio.run(main())

# import requests
# import time

# start = time.time()
# r = requests.get("https://api.ollama.cloud/v1/models")
# print("Time:", time.time() - start)
# print(r.status_code)