# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 13:45:57 2019

@author: svc_ccg
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 13:04:34 2019

@author: svc_ccg
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:29:39 2018

@author: svc_ccg
"""
import decimal  # bleh
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import url
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session
from contextlib import contextmanager

ADAPTER = "sqlserver"
DATASERVER = "AILABTRACKS\SQLEXPRESS"
USERNAME = "limstracks"
PASSWORD = "m0usetr@ck"
DATABASE = "LabTracks_AIBS"
TABLE = "Animals"


def row_to_dict(row):
    """Convert sqlalchemy table row, typically returned via a query on a table
    model, into a typical python dictionary. Each key/value pair has the
    following format:
        <row dictionary>[<column name>] = <column value for row>.

    Args:
        row (sqlalchemy row instance): Sqlalchemy row object resulting from
            a table model query. Its name will be based on the name of the table
            model.

    Returns:
        ret_dict (dict or None): Result of converting 'row' to a dictonary.
            Returns None if 'row' is 'Nonetype'.
    """
    if row is None:
        return {}
    else:
        return {
            column.name: getattr(row, column.name)
            for column in row.__table__.columns
        }
    
    
    
def table_model_factory(engine, table_name, class_name, primary_index):
    """Dynamically automap a sql table as a `sqlalchemy.schema.Table`. This
    doesn't need to exist, but it reduces a lot of boilerplate sometimes...

    Parameters
    ----------
    engine : `sqlalchemy.engine.Engine`
        `engine` used to interface with database connection/behavior
    table_name : string
        name of the table to automap
    class_name : string
        name of the `Table` schema class instance generated
    primary_index : string
        name of the column that serves as the "primary key" in the database,
        this is required by sqlalchemy regardless of whether or not it is
        required by your specific sql dialect/version/etc

    Returns
    -------
    `sqlalchemy.schema.Table`
        schema class instance representing the target table in an ORM-like way

    Notes
    -----
    - The reference to the `declarative_base` is maintained solely by its
        relationship with the returned table, so if the returned schema dies,
        the 'declarative_base' should also die...
    - This isn't the most efficient way to map multiple tables from a single
        `engine`, but is slightly helpful for automapping a single table and
        the performance/control loss isn't that bad
    """
    DatabaseBase = declarative_base()
    DatabaseBase.metadata.reflect(engine, only=[table_name])
    primary_key = getattr(DatabaseBase.metadata.tables[table_name].c,
                          primary_index)
    class_dict = {
        "__table__": DatabaseBase.metadata.tables[table_name],
        "__mapper_args__": {
            "primary_key": (primary_key, )  # must have a primary_key for mapping!
        }
    }
    class_bases = (DatabaseBase, )

    return type(class_name, class_bases, class_dict)
    
@contextmanager
def get_session(sessionmaker_factory):
    """Safe-ish way to handle a session within a context. Implements rollback
    on failure, commit on success, and close on exit.

    Parameters
    ----------
    sessionmaker_factory : `sqlalchemy.orm.session.sessionmaker`
        `sessionmaker` instance used to generate sessions

    Yields
    ------
    `sqlalchemy.orm.session.Session`
        sqlalchemy session used to manage ORM operations

    Notes
    -----
    - Technically this function yields an
        `sqlalchemy.orm.scoping.scoped_session` instance but I think that it is
        probably the same as returning a `Session` from our perspective
    """
    session = scoped_session(sessionmaker_factory)
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


# light wrappers to cleanup session in between usage
def _query_session(session, TableModel, filters=(), conditions=(), options=()):
    """Helper query an sqlalchemy orm session. Not really that necessary...

    Parameters
    ----------
    session : `sqlalchemy.orm.session.Session`
        sqlalchemy session used to manage ORM operations
    TableModel : `sqlalchemy.schema.Table`
        ORM of the table
    filters : iterable of `sqlalchemy.sql.elements.ColumnElement`, optional
        query filters
    conditions : iterable of `sqlalchemy.sql.elements.ColumnElement`, optional
        query "ORDER_BY" conditions

    Returns
    -------
    `sqlalchemy.orm.query.Query`
        `Query` instance representing the entries queried

    Notes
    -----
    As of 04/07/2017, this doesn't attempt to call the `order_by`
    method if `conditions` is empty. This fixed some
    query compatability issues with `mouse_info.orm.update`
    """
    results = session.query(TableModel) \
        .filter(*filters)

    if conditions:
        results = results.order_by(*conditions)

    if options:
        results = results.options(*options)

    return results


def query(
    sessionmaker_factory,
    TableModel,
    filters=(),
    conditions=(),
    options=(),
    adapter=row_to_dict
):
    """Create an sqlalchemy `Session` instance and query it in a somewhat
    "safe-ish" way.

    Parameters
    ----------
    sessionmaker_factory : `sqlalchemy.orm.session.sessionmaker`
        `sessionmaker` instance used to generate sessions
    filters : iterable of `sqlalchemy.sql.elements.ColumnElement`, default=()
        query filters
    conditions : iterable of `sqlalchemy.sql.elements.ColumnElement`, default=()
        query "ORDER_BY" conditions
    options : iterable of `sqlalchemy.orm.Load` interface options, default=()
        loader options supplied to the `options` method of the
        `sqlalchemy.orm.query.Query` instance
    adapter : callable, default=`mouse_info.sql.utils.row_to_dict`
        function used to convert query results into dictionaries independent
        of the original `sqlalchemy.orm.query.Query` instance

    Returns
    -------
    list of dictionaries
        with each dictionary representing a row returned by the query

    Notes
    -----
    - returning dictionaries is costly on both space and time but this function
    isn't built for speed or memory efficiency
    - we could have returned the direct result of a `sqlalchemy.orm.query.Query`
    but im dumb and flushing in `get_session` so...that won't work...>.>
    """
    with get_session(sessionmaker_factory) as session:

        q = _query_session(session, TableModel, filters, conditions, options)

        return [
            adapter(row)
            for row in q.all()
        ]    



def __fix_id(mouse_id):
    """Attempts to "fix" mouse_id if its supplied as a string.

    Parameters
    ----------
    mouse_id : integer or string
        supposed id of the mouse

    Returns
    -------
    int
        mouse id in the expected datatype/format
    """
    if isinstance(mouse_id, str):
        return int( mouse_id.lower().lstrip('m') )
    else:
        return mouse_id


def __convert_entry_dict(entry_dict):
    """Converts certain objects to more "friendly" object? like
    'decimal.Decimal' to 'float'. CAUTION: modifies the original dictionary
    passed to it.

    Notes
    -----
    - Modifies the dictionary supplied
    """
    for key, value in entry_dict.items():
        if isinstance(value, decimal.Decimal):
            entry_dict[key] = float(value)

            
def get_labtracks_animals_entry(mouse_id):
    """Fetches entry from labtracks 'Animals' table at 'ID' 'mouse_id'. Will
    attempt to "fix" the id if it gets a string value for 'mouse_id'.

    Parameters
    ----------
    mouse_id : integer or string
        'ID' to use to query the 'Animals' table.

    Returns
    -------
    dictionary
        Dictionary representation of the column returned by the query. Is in
        the format:
            <column name>: <column value>

    Notes
    -----
    - string mouse_id values are expected as an integer string or `M` prepended
    integer string
    """
    connection_url = url.URL(
        drivername="mssql+pymssql",
        username=USERNAME,
        password=PASSWORD,
        database=DATABASE,
        host=DATASERVER
    )

    _engine = create_engine(connection_url, convert_unicode=True, echo=False)
    
    LabTracks_AIBSSession = sessionmaker(bind=_engine)  # awkward name lol
    
    AnimalsModel = table_model_factory(
        _engine, "Animals", "AnimalsModel", "ID"
    )

    
    filters = [
        AnimalsModel.ID == __fix_id(mouse_id),
    ]

    conditions = []

    try:
        entry_dict = query(
            LabTracks_AIBSSession, AnimalsModel, filters, conditions
        )[0]  # there should only be one
    except IndexError:
        entry_dict = {}

    __convert_entry_dict(entry_dict)

    return entry_dict
        
        