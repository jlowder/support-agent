# specification for data generation

1. data is hierarchical with customers at the highest level, then orders, then items within orders.

2. customers have a unique ID, a name, a unique email address, a mailing address, and a loyalty tier (standard, silver, gold). They also have a list of orders, which comprise their order history.

3. Orders have a unique ID, an order date, a total amount, a shipping address, a status (pending, processing, delivered, shipped, delivered, cancelled). They also have a list of items within the order.

4. Items have a unique ID, a name, a category, a quantity, a unit price, a type (physical or digital), an is_opened boolean flag, and a list of return requests. 

5. Return requests have an item index, a request date, a reason, a status, a refund date, a refund amount, a transation id, and a boolean flag for restocking_fee_applied.

6. The data generator should generate 15 customers by default.

7. The data generator should allow the user to override the default of 15 with a -n flag. If additional users are being generated, it should an OpenAI compatible interface to interact with a LLM for generation of datafields such as customer names. Environment variables should be consulted.
	7a. LLM_URL for the URL of the OpenAI compatible interface.
	7b. LLM_MODEL for the name of the model to use.
	7c. LLM_API_KEY for the API key to use in the request.

