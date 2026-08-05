import mysql.connector

try:
    # Connect to MySQL
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="suman1719",
        database="analystbuddy"
    )

    if connection.is_connected():
        print("✅ Connected to MySQL successfully!")

except mysql.connector.Error as err:
    print("❌ Error:", err)

finally:
    if 'connection' in locals() and connection.is_connected():
        connection.close()
        print("Connection closed.")