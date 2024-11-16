from biv.dev.dto import Payment


BASE_MODEL_NAME = 'Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R-21-09-24'

_SYS_PROMPT = r'''
# Task

You have to classify a payment into a category.
Each payment has specific attributes:
* date - when a payment was created
* sum - payment amount in rubles
* description - a text that describes this payment

# Categories

* BANK_SERVICE — банковские услуги: выдача и оплата кредитов, банковские комиссии и сборы.
* FOOD_GOODS — Продовольственные товары.
* NON_FOOD_GOODS — Непродовольственные товары.
* LEASING — лизинг (финансовая аренда).
* LOAN — займы.
* REALE_STATE — недвижимость: покупка, аренда помещения, долевое инвестирование
в жилищное строительство, средства водного транспорта (не входят ЖКУ, гостиницы).
* SERVICE — услуги.
* TAX - налоги, штрафы, иные (не банковские) комиссии и сборы, социальные платежи включая
заработную плату.
* NOT_CLASSIFIED — не подходит под предыдущие категории

# Output format

Output only a category name, without anything else.
'''.strip()


def make_prompt(payment: Payment) -> list[dict[str, str]]:
    return [
        {
            'content': _SYS_PROMPT,
            'role': 'system'
        },
        {
            'content': f'Date: {payment.date}\nSum: {payment.sum}\nDescription: {payment.description}',
            'role': 'user'
        }
    ]


def make_prompt_train(payment: Payment) -> list[dict[str, str]]:
    return make_prompt(payment) + [{'role': 'assistant', 'content': payment.category}]
