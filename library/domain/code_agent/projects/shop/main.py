"""Online shop in the terminal: log in, fill a cart, check out."""

from cart import Cart
from customers import count_order, log_in
from menu import print_menu, read_choice
from orders import cancel_order, order_total, place_order
from products import products_in
from receipts import print_receipt
from reviews import average_rating, stars_text


def browse():
    for product in products_in(input("Category: ").strip().lower()):
        print(f"  {product.product_id}  {product.name:<20} {product.price:>7.2f}  {stars_text(average_rating(product.product_id))}")


def check_out(email, cart):
    code = input("Discount code (or Enter): ").strip()
    express = input("Express delivery? (y/n) ").strip().lower() == "y"
    print(f"To pay: {order_total(cart, code, express):.2f}")
    number = place_order(email, cart, code, express)
    if number is None:
        print("Sorry, something is out of stock.")
        return
    print_receipt(number, cart, code)
    count_order(email)
    cart.items.clear()


def main():
    email = input("Email: ").strip().lower()
    if log_in(email, input("Password: ")) is None:
        print("Wrong email or password.")
        return
    cart = Cart()
    while True:
        print_menu(sum(cart.items.values()))
        choice = read_choice()
        if choice == "q":
            break
        if choice == "b":
            browse()
        elif choice == "a":
            print(cart.add_item(input("Product id: ").strip(), int(input("How many? "))) or "Added.")
        elif choice == "r":
            cart.remove_item(input("Product id: ").strip())
        elif choice == "c":
            check_out(email, cart)
        elif choice == "x":
            print("Cancelled." if cancel_order(int(input("Order number: "))) else "That order cannot be cancelled.")


if __name__ == "__main__":
    main()
