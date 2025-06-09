import sqlite3
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

# Optional imports for different database types
try:
    import psycopg2
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import pymongo
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TableInfo:
    """Information about a database table"""
    name: str
    columns: List[Dict[str, Any]]
    row_count: int
    primary_keys: List[str]
    foreign_keys: List[Dict[str, str]]

@dataclass
class DatabaseSchema:
    """Complete database schema information"""
    database_type: str
    database_name: str
    tables: List[TableInfo]
    views: List[str]

class DatabaseConnector(ABC):
    """Abstract base class for database connectors"""

    @abstractmethod
    def connect(self, connection_params: Dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def get_schema(self) -> DatabaseSchema:
        pass

    @abstractmethod
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_table_data(self, table_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def close(self):
        pass

class SQLiteConnector(DatabaseConnector):
    """SQLite database connector"""

    def __init__(self):
        self.connection = None
        self.database_path = None

    def connect(self, connection_params: Dict[str, Any]) -> bool:
        try:
            self.database_path = connection_params.get('database', ':memory:')
            self.connection = sqlite3.connect(self.database_path)
            self.connection.row_factory = sqlite3.Row
            logger.info(f"Connected to SQLite database: {self.database_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to SQLite: {e}")
            return False

    def get_schema(self) -> DatabaseSchema:
        tables = []
        cursor = self.connection.cursor()

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = [row[0] for row in cursor.fetchall()]

        for table_name in table_names:
            # Get column information
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = []
            primary_keys = []

            for col_info in cursor.fetchall():
                col_dict = {
                    'name': col_info[1],
                    'type': col_info[2],
                    'nullable': not col_info[3],
                    'default': col_info[4]
                }
                columns.append(col_dict)
                if col_info[5]:  # Primary key
                    primary_keys.append(col_info[1])

            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]

            # Get foreign keys
            cursor.execute(f"PRAGMA foreign_key_list({table_name})")
            foreign_keys = []
            for fk_info in cursor.fetchall():
                foreign_keys.append({
                    'column': fk_info[3],
                    'referenced_table': fk_info[2],
                    'referenced_column': fk_info[4]
                })

            table_info = TableInfo(
                name=table_name,
                columns=columns,
                row_count=row_count,
                primary_keys=primary_keys,
                foreign_keys=foreign_keys
            )
            tables.append(table_info)

        # Get views
        cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
        views = [row[0] for row in cursor.fetchall()]

        return DatabaseSchema(
            database_type='SQLite',
            database_name=self.database_path,
            tables=tables,
            views=views
        )

    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        cursor = self.connection.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)

        columns = [description[0] for description in cursor.description] if cursor.description else []
        rows = cursor.fetchall()

        return [dict(zip(columns, row)) for row in rows]

    def get_table_data(self, table_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        query = f"SELECT * FROM {table_name} LIMIT ?"
        return self.execute_query(query, (limit,))

    def close(self):
        if self.connection:
            self.connection.close()
            logger.info("SQLite connection closed")

class PostgreSQLConnector(DatabaseConnector):
    """PostgreSQL database connector"""

    def __init__(self):
        self.connection = None
        self.database_name = None

    def connect(self, connection_params: Dict[str, Any]) -> bool:
        if not POSTGRES_AVAILABLE:
            logger.error("psycopg2 not installed. Install with: pip install psycopg2-binary")
            return False

        try:
            self.connection = psycopg2.connect(**connection_params)
            self.database_name = connection_params.get('database', 'unknown')
            logger.info(f"Connected to PostgreSQL database: {self.database_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            return False

    def get_schema(self) -> DatabaseSchema:
        cursor = self.connection.cursor()
        tables = []

        # Get all tables in public schema
        cursor.execute("""
                       SELECT table_name
                       FROM information_schema.tables
                       WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                       """)
        table_names = [row[0] for row in cursor.fetchall()]

        for table_name in table_names:
            # Get column information
            cursor.execute("""
                           SELECT column_name, data_type, is_nullable, column_default
                           FROM information_schema.columns
                           WHERE table_name = %s AND table_schema = 'public'
                           ORDER BY ordinal_position
                           """, (table_name,))

            columns = []
            for col_info in cursor.fetchall():
                columns.append({
                    'name': col_info[0],
                    'type': col_info[1],
                    'nullable': col_info[2] == 'YES',
                    'default': col_info[3]
                })

            # Get primary keys
            cursor.execute("""
                           SELECT column_name
                           FROM information_schema.key_column_usage
                           WHERE table_name = %s AND constraint_name IN (
                               SELECT constraint_name
                               FROM information_schema.table_constraints
                               WHERE table_name = %s AND constraint_type = 'PRIMARY KEY'
                           )
                           """, (table_name, table_name))
            primary_keys = [row[0] for row in cursor.fetchall()]

            # Get row count
            cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
            row_count = cursor.fetchone()[0]

            table_info = TableInfo(
                name=table_name,
                columns=columns,
                row_count=row_count,
                primary_keys=primary_keys,
                foreign_keys=[]  # Simplified for this example
            )
            tables.append(table_info)

        # Get views
        cursor.execute("""
                       SELECT table_name
                       FROM information_schema.views
                       WHERE table_schema = 'public'
                       """)
        views = [row[0] for row in cursor.fetchall()]

        return DatabaseSchema(
            database_type='PostgreSQL',
            database_name=self.database_name,
            tables=tables,
            views=views
        )

    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        cursor = self.connection.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)

        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()

        return [dict(zip(columns, row)) for row in rows]

    def get_table_data(self, table_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        query = f'SELECT * FROM "{table_name}" LIMIT %s'
        return self.execute_query(query, (limit,))

    def close(self):
        if self.connection:
            self.connection.close()
            logger.info("PostgreSQL connection closed")

class UniversalDatabaseExplorer:
    """Main class for universal database exploration"""

    def __init__(self):
        self.connector = None
        self.schema = None
        self.supported_databases = {
            'sqlite': SQLiteConnector,
            'postgresql': PostgreSQLConnector,
            # Ready to implement:
            # 'mysql': MySQLConnector,
            # 'mongodb': MongoDBConnector,
            # 'mssql': SQLServerConnector,
            # 'oracle': OracleConnector,
            # 'redis': RedisConnector,
            # 'cassandra': CassandraConnector,
            # 'elasticsearch': ElasticsearchConnector,
        }

    def get_supported_databases(self) -> dict:
        """Get list of all supported database types and their requirements"""
        return {
            'sqlite': {
                'status': 'implemented',
                'requirements': 'built-in (no extra packages needed)',
                'connection_params': ['database (file path)']
            },
            'postgresql': {
                'status': 'implemented',
                'requirements': 'pip install psycopg2-binary',
                'connection_params': ['host', 'port', 'database', 'user', 'password']
            },
            'mysql': {
                'status': 'framework_ready',
                'requirements': 'pip install mysql-connector-python',
                'connection_params': ['host', 'port', 'database', 'user', 'password']
            },
            'mongodb': {
                'status': 'framework_ready',
                'requirements': 'pip install pymongo',
                'connection_params': ['host', 'port', 'database', 'username', 'password']
            },
            'mssql': {
                'status': 'can_implement',
                'requirements': 'pip install pyodbc',
                'connection_params': ['server', 'database', 'username', 'password', 'driver']
            },
            'oracle': {
                'status': 'can_implement',
                'requirements': 'pip install cx_Oracle',
                'connection_params': ['host', 'port', 'service_name', 'user', 'password']
            },
            'redis': {
                'status': 'can_implement',
                'requirements': 'pip install redis',
                'connection_params': ['host', 'port', 'db', 'password']
            },
            'cassandra': {
                'status': 'can_implement',
                'requirements': 'pip install cassandra-driver',
                'connection_params': ['hosts', 'keyspace', 'username', 'password']
            },
            'elasticsearch': {
                'status': 'can_implement',
                'requirements': 'pip install elasticsearch',
                'connection_params': ['hosts', 'index', 'username', 'password']
            }
        }

    def connect(self, db_type: str, connection_params: Dict[str, Any]) -> bool:
        """Connect to a database"""
        db_type = db_type.lower()

        if db_type not in self.supported_databases:
            logger.error(f"Unsupported database type: {db_type}")
            logger.info(f"Supported types: {list(self.supported_databases.keys())}")
            return False

        connector_class = self.supported_databases[db_type]
        self.connector = connector_class()

        if self.connector.connect(connection_params):
            self.schema = self.connector.get_schema()
            return True
        return False

    def get_database_info(self) -> Dict[str, Any]:
        """Get complete database information"""
        if not self.schema:
            return {}

        return {
            'database_type': self.schema.database_type,
            'database_name': self.schema.database_name,
            'total_tables': len(self.schema.tables),
            'total_views': len(self.schema.views),
            'tables': [
                {
                    'name': table.name,
                    'columns': len(table.columns),
                    'rows': table.row_count,
                    'primary_keys': table.primary_keys
                }
                for table in self.schema.tables
            ]
        }

    def get_table_structure(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed structure of a specific table"""
        if not self.schema:
            return None

        for table in self.schema.tables:
            if table.name == table_name:
                return {
                    'name': table.name,
                    'columns': table.columns,
                    'row_count': table.row_count,
                    'primary_keys': table.primary_keys,
                    'foreign_keys': table.foreign_keys
                }
        return None

    def query_data(self, query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """Execute a custom query"""
        if not self.connector:
            return []
        return self.connector.execute_query(query, params)

    def get_sample_data(self, table_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get sample data from a table"""
        if not self.connector:
            return []
        return self.connector.get_table_data(table_name, limit)

    def search_tables(self, keyword: str) -> List[str]:
        """Search for tables containing a keyword"""
        if not self.schema:
            return []

        matching_tables = []
        keyword = keyword.lower()

        for table in self.schema.tables:
            if keyword in table.name.lower():
                matching_tables.append(table.name)
            else:
                # Check column names
                for column in table.columns:
                    if keyword in column['name'].lower():
                        matching_tables.append(table.name)
                        break

        return matching_tables

    def get_related_tables(self, table_name: str) -> List[str]:
        """Find tables related through foreign keys"""
        if not self.schema:
            return []

        related = []
        target_table = None

        # Find the target table
        for table in self.schema.tables:
            if table.name == table_name:
                target_table = table
                break

        if not target_table:
            return []

        # Find tables referenced by foreign keys
        for fk in target_table.foreign_keys:
            if fk['referenced_table'] not in related:
                related.append(fk['referenced_table'])

        # Find tables that reference this table
        for table in self.schema.tables:
            for fk in table.foreign_keys:
                if fk['referenced_table'] == table_name and table.name not in related:
                    related.append(table.name)

        return related

    def close(self):
        """Close database connection"""
        if self.connector:
            self.connector.close()

# Example usage and demo
def demo_usage():
    """Demonstrate how to use the Universal Database Explorer"""

    # Create explorer instance
    explorer = UniversalDatabaseExplorer()

    print("=== Universal Database Explorer Demo ===")

    # Example 1: Connect to any existing database
    database_files = [
        'student_database.db',
        'sales_database.db',
        'hr_database.db',
        'inventory.db'
    ]

    print("\nTrying to connect to different databases...")

    for db_file in database_files:
        print(f"\n--- Attempting to connect to {db_file} ---")

        if explorer.connect('sqlite', {'database': db_file}):
            print(f"✓ Connected successfully to {db_file}!")

            # Get database overview
            db_info = explorer.get_database_info()
            print(f"Database Type: {db_info.get('database_type')}")
            print(f"Total Tables: {db_info.get('total_tables')}")
            print(f"Total Views: {db_info.get('total_views')}")

            # List all tables
            if db_info.get('tables'):
                print("\nTables found:")
                for table in db_info['tables']:
                    print(f"  - {table['name']}: {table['columns']} columns, {table['rows']} rows")

            # Demonstrate dynamic exploration
            if db_info.get('tables'):
                first_table = db_info['tables'][0]['name']
                print(f"\nExploring table '{first_table}':")

                # Get table structure
                structure = explorer.get_table_structure(first_table)
                if structure:
                    print("Columns:")
                    for col in structure['columns']:
                        print(f"  - {col['name']} ({col['type']})")

                # Get sample data
                sample = explorer.get_sample_data(first_table, limit=3)
                if sample:
                    print(f"\nSample data (first 3 rows):")
                    for i, row in enumerate(sample, 1):
                        print(f"  Row {i}: {dict(row)}")

            explorer.close()
            print(f"Connection to {db_file} closed.\n")

        else:
            print(f"✗ Could not connect to {db_file} (file may not exist)")

    # Example 2: Create and explore a sample student database
    print("\n=== Creating Sample Student Database ===")
    create_sample_student_db()

    if explorer.connect('sqlite', {'database': 'sample_student_database.db'}):
        print("Connected to sample student database!")

        # Demonstrate search functionality
        student_tables = explorer.search_tables('student')
        print(f"Tables containing 'student': {student_tables}")

        course_tables = explorer.search_tables('course')
        print(f"Tables containing 'course': {course_tables}")

        # Show relationships
        if student_tables:
            related = explorer.get_related_tables(student_tables[0])
            print(f"Tables related to '{student_tables[0]}': {related}")

        # Custom queries
        print("\nCustom query examples:")

        # Find students with high GPA
        high_gpa = explorer.query_data("SELECT * FROM students WHERE gpa > ?", (3.5,))
        print(f"Students with GPA > 3.5: {len(high_gpa)} found")

        # Count students by major
        major_counts = explorer.query_data("""
                                           SELECT major, COUNT(*) as count
                                           FROM students
                                           GROUP BY major
                                           ORDER BY count DESC
                                           """)
        print("Students by major:")
        for row in major_counts:
            print(f"  {row['major']}: {row['count']} students")

        explorer.close()

def create_sample_student_db():
    """Create a sample student database for demonstration"""
    conn = sqlite3.connect('sample_student_database.db')
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS students (
                                                           student_id INTEGER PRIMARY KEY,
                                                           name TEXT NOT NULL,
                                                           email TEXT UNIQUE,
                                                           major TEXT,
                                                           gpa REAL,
                                                           enrollment_date TEXT
                   )
                   ''')

    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS courses (
                                                          course_id INTEGER PRIMARY KEY,
                                                          course_name TEXT NOT NULL,
                                                          credits INTEGER,
                                                          instructor TEXT,
                                                          department TEXT
                   )
                   ''')

    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS enrollments (
                                                              enrollment_id INTEGER PRIMARY KEY,
                                                              student_id INTEGER,
                                                              course_id INTEGER,
                                                              grade TEXT,
                                                              semester TEXT,
                                                              FOREIGN KEY (student_id) REFERENCES students(student_id),
                       FOREIGN KEY (course_id) REFERENCES courses(course_id)
                       )
                   ''')

    # Insert sample data
    students_data = [
        (1, 'Alice Johnson', 'alice@university.edu', 'Computer Science', 3.8, '2023-09-01'),
        (2, 'Bob Smith', 'bob@university.edu', 'Mathematics', 3.6, '2023-09-01'),
        (3, 'Carol Davis', 'carol@university.edu', 'Physics', 3.9, '2023-09-01'),
        (4, 'David Wilson', 'david@university.edu', 'Computer Science', 3.4, '2023-09-01'),
    ]

    courses_data = [
        (1, 'Introduction to Programming', 3, 'Dr. Brown', 'Computer Science'),
        (2, 'Calculus I', 4, 'Prof. Adams', 'Mathematics'),
        (3, 'Physics I', 4, 'Dr. Clark', 'Physics'),
        (4, 'Data Structures', 3, 'Dr. Brown', 'Computer Science'),
    ]

    enrollments_data = [
        (1, 1, 1, 'A', 'Fall 2023'),
        (2, 1, 4, 'A-', 'Fall 2023'),
        (3, 2, 2, 'B+', 'Fall 2023'),
        (4, 3, 3, 'A', 'Fall 2023'),
        (5, 4, 1, 'B', 'Fall 2023'),
    ]

    cursor.executemany('INSERT OR REPLACE INTO students VALUES (?, ?, ?, ?, ?, ?)', students_data)
    cursor.executemany('INSERT OR REPLACE INTO courses VALUES (?, ?, ?, ?, ?)', courses_data)
    cursor.executemany('INSERT OR REPLACE INTO enrollments VALUES (?, ?, ?, ?, ?)', enrollments_data)

    conn.commit()
    conn.close()
    print("Sample student database created: sample_student_database.db")

def explore_any_database(database_path: str):
    """Helper function to explore any database file"""
    explorer = UniversalDatabaseExplorer()

    print(f"=== Exploring {database_path} ===")

    if explorer.connect('sqlite', {'database': database_path}):
        db_info = explorer.get_database_info()

        print(f"Database: {db_info['database_name']}")
        print(f"Tables: {db_info['total_tables']}")
        print(f"Views: {db_info['total_views']}")

        # List all tables with details
        for table in db_info['tables']:
            print(f"\nTable: {table['name']}")
            print(f"  Columns: {table['columns']}")
            print(f"  Rows: {table['rows']}")

            # Show column details
            structure = explorer.get_table_structure(table['name'])
            if structure:
                print("  Column details:")
                for col in structure['columns']:
                    nullable = "NULL" if col['nullable'] else "NOT NULL"
                    print(f"    {col['name']}: {col['type']} {nullable}")

        explorer.close()
        return True
    else:
        print(f"Could not connect to {database_path}")
        return False

def real_world_examples():
    """Show real-world usage scenarios - no manual setup required"""

    explorer = UniversalDatabaseExplorer()

    print("=== Real-World Database Connection Examples ===\n")

    # Scenario 1: Legacy SQLite files
    print("1. EXISTING SQLITE FILES (no setup needed)")
    sqlite_examples = [
        "student_records.db",      # School management system
        "sales_data.db",           # Point of sale system
        "inventory.db",            # Warehouse management
        "customers.sqlite3",       # CRM system export
        "financial_data.db",       # Accounting software
        "logs.db"                  # Application logs
    ]

    for db_file in sqlite_examples:
        print(f"  explorer.connect('sqlite', {{'database': '{db_file}'}})")

    # Scenario 2: Production PostgreSQL
    print("\n2. PRODUCTION POSTGRESQL SERVERS (no migration needed)")
    print("  # Company's main HR database")
    print("  explorer.connect('postgresql', {")
    print("      'host': 'hr-db.company.com',")
    print("      'database': 'human_resources',")
    print("      'user': 'readonly_user',")
    print("      'password': 'secure_password'")
    print("  })")

    print("\n  # E-commerce database")
    print("  explorer.connect('postgresql', {")
    print("      'host': 'prod-db.ecommerce.com',")
    print("      'database': 'online_store',")
    print("      'user': 'analyst',")
    print("      'password': 'analyst_pass'")
    print("  })")

    # Scenario 3: Local development databases
    print("\n3. LOCAL DEVELOPMENT DATABASES")
    print("  # Local MySQL for testing")
    print("  explorer.connect('mysql', {")
    print("      'host': 'localhost',")
    print("      'database': 'test_app',")
    print("      'user': 'developer',")
    print("      'password': 'dev_password'")
    print("  })")

    # Scenario 4: Cloud databases
    print("\n4. CLOUD DATABASES (direct connection)")
    print("  # AWS RDS PostgreSQL")
    print("  explorer.connect('postgresql', {")
    print("      'host': 'myapp.abcd1234.us-east-1.rds.amazonaws.com',")
    print("      'database': 'production',")
    print("      'user': 'app_user',")
    print("      'password': 'cloud_password'")
    print("  })")

    print("\n  # Google Cloud SQL")
    print("  explorer.connect('mysql', {")
    print("      'host': '10.1.2.3',  # Cloud SQL private IP")
    print("      'database': 'analytics',")
    print("      'user': 'data_analyst',")
    print("      'password': 'gcp_password'")
    print("  })")

def demonstrate_no_setup_needed():
    """Show that the app works with databases as-is"""

    print("=== NO MANUAL SETUP REQUIRED ===\n")

    print("✅ WHAT THE APP DOES:")
    print("  • Connects to existing databases in their current location")
    print("  • Automatically discovers tables, columns, and relationships")
    print("  • Reads data directly from the original database")
    print("  • Works with multiple database types simultaneously")
    print("  • No data copying or migration required")

    print("\n❌ WHAT YOU DON'T NEED TO DO:")
    print("  • Don't copy .db files to other database systems")
    print("  • Don't convert between database formats")
    print("  • Don't set up new database servers")
    print("  • Don't import/export data")
    print("  • Don't modify existing database structures")

    print("\n🎯 TYPICAL WORKFLOW:")
    print("  1. Point the app to your existing database")
    print("  2. App automatically maps the entire structure")
    print("  3. Start exploring and querying immediately")
    print("  4. Switch to different databases as needed")

    print("\n📂 EXAMPLE: Using Existing Company Databases")
    explorer = UniversalDatabaseExplorer()

    # Show how you'd connect to various existing systems
    databases_in_company = {
        'HR System': ('sqlite', {'database': '/company/hr/employee_data.db'}),
        'Sales CRM': ('postgresql', {'host': 'crm-server', 'database': 'sales', 'user': 'analyst', 'password': 'pass'}),
        'Inventory': ('mysql', {'host': 'warehouse-db', 'database': 'inventory', 'user': 'readonly', 'password': 'pass'}),
        'Customer Support': ('sqlite', {'database': '/support/tickets.db'}),
    }

    for system_name, (db_type, connection_params) in databases_in_company.items():
        print(f"\n  Connecting to {system_name}:")
        print(f"    explorer.connect('{db_type}', {connection_params})")
        print(f"    # Instantly access all tables and data from {system_name}")

def multi_database_exploration():
    """Show how to work with multiple databases simultaneously"""

    print("\n=== WORKING WITH MULTIPLE DATABASES ===")

    print("""
# You can explore multiple databases in the same session:

# 1. Check the student database
explorer.connect('sqlite', {'database': 'student_database.db'})
student_info = explorer.get_database_info()
high_gpa_students = explorer.query_data("SELECT * FROM students WHERE gpa > 3.5")
explorer.close()

# 2. Switch to sales database  
explorer.connect('sqlite', {'database': 'sales_database.db'})
sales_info = explorer.get_database_info()
top_products = explorer.query_data("SELECT product, SUM(revenue) FROM sales GROUP BY product")
explorer.close()

# 3. Connect to production PostgreSQL
explorer.connect('postgresql', {
    'host': 'prod-server.com',
    'database': 'company_data', 
    'user': 'analyst',
    'password': 'secure_pass'
})
company_metrics = explorer.query_data("SELECT department, COUNT(*) FROM employees GROUP BY department")
explorer.close()

# No data migration between these systems - each stays in its original format!
""")

if __name__ == "__main__":
    real_world_examples()
    demonstrate_no_setup_needed()
    multi_database_exploration()