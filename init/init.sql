CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        email VARCHAR(100) UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
        
CREATE TABLE IF NOT EXISTS products (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            price DECIMAL(10,2),
            category VARCHAR(50),
            stock INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
INSERT INTO users (name, email) VALUES 
        ('Max Mustermann', 'max@example.com'),
        ('Anna Schmidt', 'anna@example.com'),
        ('Tom Weber', 'tom@example.com')
        ON CONFLICT (email) DO NOTHING;
        
        INSERT INTO products (name, price, category, stock) VALUES 
        ('Laptop', 999.99, 'Electronics', 10),
        ('Mouse', 29.99, 'Electronics', 50),
        ('Book', 19.99, 'Education', 100),
        ('Chair', 149.99, 'Furniture', 15)
        ON CONFLICT DO NOTHING;