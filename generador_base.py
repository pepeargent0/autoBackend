import datetime
import decimal
import importlib
import inspect
import inspect as _inspect
import json
import os
import pathlib
import pathlib as _pl
import re
import re as _re
import shlex
import subprocess
import tempfile
import uuid
import yaml
from textwrap import dedent
from typing import Any
from typing import Dict, List, Tuple, Set,Optional
from typing import Iterable
from pathlib import Path

import typer
from dotenv import load_dotenv

app = typer.Typer()
_CTRL_SENSITIVE_RE = _re.compile(r"(pass|password|clave|secret|token|salt|hash)", _re.I)
_CTRL_FILE_RE = _re.compile(r"(logo|imagen|foto|avatar|file|archivo|pdf|doc|path|url)", _re.I)
def _ctrl_pk_name(model_cls) -> str | None:
    try:
        pks = [c.name for c in model_cls.__table__.primary_key]  # type: ignore
        return pks[0] if pks else None
    except Exception:
        return None
def _ctrl_columns(model_cls):
    try:
        return list(model_cls.__table__.columns)  # type: ignore
    except Exception:
        return []
def _ctrl_string_cols(model_cls, limit=5):
    cols = []
    for c in _ctrl_columns(model_cls):
        t = c.type.__class__.__name__.lower()
        if (("string" in t) or ("text" in t) or ("varchar" in t) or ("citext" in t)) and not _CTRL_SENSITIVE_RE.search(c.name):
            cols.append(c.name)
    pri = ["empresa","nombre","apellido","email","telefono","provincia","ciudad","descripcion","titulo"]
    cols = sorted(cols, key=lambda n: (0 if n in pri else 1, n))
    return cols[:limit] if cols else []
def _ctrl_file_cols(model_cls):
    return [c.name for c in _ctrl_columns(model_cls) if _CTRL_FILE_RE.search(c.name)]
def _ctrl_non_sensitive(model_cls):
    return [c.name for c in _ctrl_columns(model_cls) if not _CTRL_SENSITIVE_RE.search(c.name)]
def _to_camel(s: str) -> str:
    return "".join(p.capitalize() for p in s.split("_"))
def _singular(resource: str) -> str:
    return resource[:-1] if resource.endswith("s") else resource
def _singular_guess(name: str) -> str:
    if name.endswith("es"):
        return name[:-2]
    if name.endswith("s"):
        return name[:-1]
    return name
def _find_model_class(models_pkg: str, resource: str):
    modelos = importlib.import_module(models_pkg)
    candidates = [
        _to_camel(resource),
        _to_camel(_singular_guess(resource)),
    ]
    # recorro clases que hereden de Base
    Base = getattr(modelos, "Base", None)
    classes = [obj for _, obj in _inspect.getmembers(modelos) if _inspect.isclass(obj)]
    for cand in candidates:
        for cls in classes:
            if cls.__name__ == cand:
                if Base and issubclass(cls, Base):
                    return cls
    # fallback: primera clase “no Base”
    for cls in classes:
        if Base and cls is not Base and issubclass(cls, Base):
            return cls
    return None
SERVICE_OVERRIDES = {
    "empresas":  "UsuariosService",
    "clientes":  "UsuariosService",
    "empleados": "UsuariosService",
    "encargados":"UsuariosService",
    "vendedores":"UsuariosService",
}
PARAM_OVERRIDES = {
    "empresas": "id_cliente",
    "clientes": "id_cliente",
}
def _camelize(s: str) -> str:
    return "".join(p.capitalize() for p in s.split("_"))
def _try_pick_class(services_pkg: str, candidates: list[str]) -> str | None:
    try:
        mod = importlib.import_module(services_pkg)
    except Exception:
        return None
    for cls in candidates:
        if hasattr(mod, cls):
            return cls
    return None
def service_class_for(resource: str, role: str, services_pkg: str) -> str:
    """
    1) usa overrides (empresas->UsuariosService, etc.)
    2) intenta candidatos por convención y valida que existan en `services_pkg`
    3) fallback: UsuariosService si existe, si no, último candidato camelizado
    """
    # 1) override explícito
    override = SERVICE_OVERRIDES.get(resource)
    if override:
        cls = _try_pick_class(services_pkg, [override])
        if cls:
            return cls

    # 2) convenciones habituales
    sing = _singular(resource)
    candidates = [
        f"{_camelize(resource)}Service",   # p.ej. abono_mensual -> AbonoMensualService
        f"{_camelize(sing)}Service",       # empresas -> EmpresaService
        f"{resource.capitalize()}Service", # legacy simple
    ]
    cls = _try_pick_class(services_pkg, candidates)
    if cls:
        return cls

    # 3) fallback razonable a UsuariosService si existe
    cls = _try_pick_class(services_pkg, ["UsuariosService"])
    if cls:
        return cls

    # último recurso: devolver el 1ro por convención (aunque no exista) para que el dev vea el error rápido
    return candidates[0]
BASE_FILE_CONTENT = """\
from sqlalchemy.orm import DeclarativeBase
class Base(DeclarativeBase):
    pass
"""
MODEL_HEADER = """\
from __future__ import annotations
from typing import Optional, List, Dict, Any
import decimal
import datetime
from sqlalchemy import (
    Column, Integer, String, Boolean, Float, Date, DateTime, Time, Text,
    BigInteger, SmallInteger, LargeBinary, Numeric, JSON,
    ForeignKey, ForeignKeyConstraint, PrimaryKeyConstraint,
    UniqueConstraint, CheckConstraint, Index, text
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY, DOUBLE_PRECISION as _PG_DOUBLE
from .base import Base

# Compat Double(53) -> DOUBLE PRECISION en PG
def Double(*args, **kwargs):
    return _PG_DOUBLE()
"""
SENSITIVE_RE = r"(pass|password|clave|secret|token|salt|hash)"
def get_db_url_from_env() -> str:
    load_dotenv()
    server = os.getenv("DB_SERVER", "localhost")
    port = os.getenv("DB_PORT", "5432")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "")
    name = os.getenv("DB_NAME", "")
    return f"postgresql+psycopg2://{user}:{password}@{server}:{port}/{name}"
def run_sqlacodegen(db_url: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_path = tmp_file.name
    cmd = f"sqlacodegen {shlex.quote(db_url)} --outfile {shlex.quote(tmp_path)}"
    subprocess.run(shlex.split(cmd), check=True)
    return pathlib.Path(tmp_path).read_text(encoding="utf-8")
@app.command(name="generate-models", help="Genera modelos (un archivo por tabla)")
def generate_models(out_dir: str = "modelos"):
    db_url = get_db_url_from_env()
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "base.py").write_text(BASE_FILE_CONTENT, encoding="utf-8")
    typer.secho(">> Reflejando DB y generando modelos...", fg=typer.colors.CYAN)
    content = run_sqlacodegen(db_url)
    parts = re.split(r'(?m)^(class\s+\w+\(Base\):)', content)
    model_files: List[tuple[str, str]] = []
    if len(parts) == 1:
        typer.secho("⚠ No se detectaron clases en la salida de sqlacodegen.", fg=typer.colors.YELLOW)
    else:
        for i in range(1, len(parts), 2):
            class_def = parts[i].strip()
            body = parts[i + 1].rstrip() + "\n"
            m = re.match(r'class\s+(\w+)', class_def)
            if not m:
                continue
            class_name = m.group(1)
            file_path = out_path / f"{class_name.lower()}.py"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(MODEL_HEADER)
                f.write(class_def + "\n")
                f.write(body)
            model_files.append((class_name, file_path.name))
            typer.secho(f"✓ {file_path} generado", fg=typer.colors.GREEN)
    init_lines = ["# Auto-generado. Importa Base y modelos aquí.\n", "from .base import Base\n"]
    for class_name, module_name in model_files:
        mod = module_name[:-3]
        init_lines.append(f"from .{mod} import {class_name}\n")
    init_lines.append("\n__all__ = [\n    'Base',\n")
    for name, _ in model_files:
        init_lines.append(f"    '{name}',\n")
    init_lines.append("]\n")
    (out_path / "__init__.py").write_text("".join(init_lines), encoding="utf-8")
    typer.secho(f"Listo ✅ Modelos generados en {out_path}", fg=typer.colors.GREEN)
REPOS_BASE_FILE = """\
import logging
import re
from typing import Type, TypeVar, Generic, Optional, List, Any, Dict, Tuple, Iterable
from sqlalchemy import select, func, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.inspection import inspect as sa_inspect
from sqlalchemy.orm import RelationshipProperty, joinedload
from sqlalchemy.orm.attributes import InstrumentedAttribute

T = TypeVar("T")
logger = logging.getLogger(__name__)

_SENSITIVE_PATTERNS = re.compile(r"(pass|password|clave|secret|token|salt|hash)", re.IGNORECASE)

class AsyncBaseRepository(Generic[T]):
    \"\"\"
    CRUD + consultas avanzadas con:
      - PK real (no asume 'id')
      - get_by_property(prop, value) con paths 'rel.campo'
      - list_paginated: texto (modelo + relaciones) excluyendo campos sensibles,
        filtros (__gte/__lte/__gt/__lt/__in/__like/__ilike), orden, y JOINS automáticos
      - soft delete si existe 'borrado'
      - joinedload helpers por relación
    \"\"\"

    def __init__(self, model: Type[T], session: AsyncSession):
        self.model = model
        self.session = session
        self._mapper = sa_inspect(self.model)

    # ---------------------------
    # Introspección
    # ---------------------------
    def _columns_map(self) -> Dict[str, InstrumentedAttribute]:
        return {c.key: getattr(self.model, c.key) for c in self._mapper.mapper.column_attrs}

    def _pk_cols(self) -> List[InstrumentedAttribute]:
        return list(self._mapper.primary_key)

    def _pk_name(self) -> Optional[str]:
        pks = self._pk_cols()
        return pks[0].key if pks else None

    def _relationships(self) -> Dict[str, RelationshipProperty]:
        return {rel.key: rel for rel in self._mapper.relationships}

    @staticmethod
    def _is_sensitive(col_name: str) -> bool:
        return bool(_SENSITIVE_PATTERNS.search(col_name))

    def _text_columns_local(self) -> List[InstrumentedAttribute]:
        cols = []
        for c in self._mapper.columns:
            t = c.type.__class__.__name__.lower()
            if ("string" in t or "text" in t or "varchar" in t or "citext" in t) and not self._is_sensitive(c.key):
                cols.append(getattr(self.model, c.key))
        return cols

    def _related_text_columns(self, rel_name: str) -> List[Tuple[str, InstrumentedAttribute]]:
        rels = self._relationships()
        if rel_name not in rels:
            return []
        target = rels[rel_name].mapper.class_
        mapper = sa_inspect(target)
        out = []
        for c in mapper.columns:
            t = c.type.__class__.__name__.lower()
            if ("string" in t or "text" in t or "varchar" in t or "citext" in t) and not self._is_sensitive(c.key):
                out.append((f"{rel_name}.{c.key}", getattr(target, c.key)))
        return out

    # ---------------------------
    # Resolución de paths con punto (rel.campo)
    # ---------------------------
    def _resolve_path(self, path: str):
        if "." not in path:
            col = self._columns_map().get(path)
            return (None, col)
        rel_name, col_name = path.split(".", 1)
        rel = self._relationships().get(rel_name)
        if not rel:
            return (None, None)
        target = sa_inspect(rel.mapper.class_)
        try:
            return (rel_name, getattr(target.class_, col_name))
        except Exception:
            return (None, None)

    # ---------------------------
    # Builders de condiciones
    # ---------------------------
    def _build_filters(self, filters: Optional[Dict[str, Any]], joins_needed: set) -> List[Any]:
        if not filters:
            return []
        conds = []
        for key, value in filters.items():
            op = None
            col_path = key
            for suffix in ("__gte", "__lte", "__gt", "__lt", "__in", "__like", "__ilike"):
                if key.endswith(suffix):
                    col_path = key[: -len(suffix)]
                    op = suffix
                    break
            rel_name, attr = self._resolve_path(col_path)
            if attr is None:
                logger.debug("Filtro ignorado: %s no existe", col_path)
                continue
            if rel_name:
                joins_needed.add(rel_name)
            if op is None:
                conds.append(attr == value)
            elif op == "__gte":
                conds.append(attr >= value)
            elif op == "__lte":
                conds.append(attr <= value)
            elif op == "__gt":
                conds.append(attr > value)
            elif op == "__lt":
                conds.append(attr < value)
            elif op == "__in":
                seq = list(value) if isinstance(value, (list, tuple, set)) else [value]
                conds.append(attr.in_(seq))
            elif op == "__like":
                conds.append(attr.like(value))
            elif op == "__ilike":
                try:
                    conds.append(attr.ilike(value))
                except Exception:
                    conds.append(attr.like(value))
        return conds

    def _build_search(self, search: Optional[str], joins_needed: set, joins: Optional[List[str]], include_joined_in_search: bool):
        if not search:
            return None
        ors = [col.ilike(f"%{search}%") if hasattr(col, "ilike") else col.like(f"%{search}%")
               for col in self._text_columns_local()]
        if include_joined_in_search and joins:
            for rel_name in joins:
                for _qname, attr in self._related_text_columns(rel_name):
                    joins_needed.add(rel_name)
                    try:
                        ors.append(attr.ilike(f"%{search}%"))
                    except Exception:
                        ors.append(attr.like(f"%{search}%"))
        return or_(*ors) if ors else None

    def _build_order(self, order_by: Optional[str], order_dir: str, joins_needed: set):
        if not order_by:
            return None
        rel_name, attr = self._resolve_path(order_by)
        if attr is None:
            logger.debug("order_by ignorado: %s no existe", order_by)
            return None
        if rel_name:
            joins_needed.add(rel_name)
        return attr.desc() if str(order_dir).lower() == "desc" else attr.asc()

    # ---------------------------
    # Joins
    # ---------------------------
    def _apply_joins(self, stmt, joins_needed: Iterable[str]):
        rels = self._relationships()
        for rel_name in joins_needed:
            rel = rels.get(rel_name)
            if not rel:
                continue
            stmt = stmt.join(getattr(self.model, rel_name), isouter=True)
        return stmt

    # ---------------------------
    # CRUD + helpers
    # ---------------------------
    async def get_by_pk(self, pk_value: Any) -> Optional[T]:
        pk = self._pk_name()
        if not pk:
            return None
        res = await self.session.execute(select(self.model).where(getattr(self.model, pk) == pk_value))
        return res.scalars().first()

    async def get_by_id(self, entity_id: Any) -> Optional[T]:
        if "id" in self._columns_map():
            res = await self.session.execute(select(self.model).where(getattr(self.model, "id") == entity_id))
            return res.scalars().first()
        return await self.get_by_pk(entity_id)

    async def get_by_property(self, prop: str, value: Any) -> Optional[T]:
        joins_needed = set()
        rel_name, attr = self._resolve_path(prop)
        if attr is None:
            logger.warning("[get_by_property] columna '%s' no existe", prop)
            return None
        if rel_name:
            joins_needed.add(rel_name)
        stmt = select(self.model)
        stmt = self._apply_joins(stmt, joins_needed)
        stmt = stmt.where(attr == value)
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def get_all(self) -> List[T]:
        res = await self.session.execute(select(self.model))
        return res.scalars().all()

    async def create(self, data: Dict[str, Any]) -> Optional[T]:
        try:
            obj = self.model(**data)
            self.session.add(obj)
            await self.session.commit()
            await self.session.refresh(obj)
            return obj
        except Exception as e:
            await self.session.rollback()
            logger.error("[create] %s", e)
            return None

    async def update(self, entity: T, data: Dict[str, Any]) -> Optional[T]:
        try:
            for k, v in data.items():
                if hasattr(entity, k):
                    setattr(entity, k, v)
            await self.session.commit()
            return entity
        except Exception as e:
            await self.session.rollback()
            logger.error("[update] %s", e)
            return None

    async def delete(self, entity_or_pk: Any, soft_field: str = "borrado", soft_yes: str = "si") -> bool:
        try:
            obj = entity_or_pk
            if not hasattr(entity_or_pk, "__table__"):
                obj = await self.get_by_pk(entity_or_pk)
                if not obj:
                    return False
            cols = self._columns_map()
            if soft_field in cols:
                setattr(obj, soft_field, soft_yes)
            else:
                self.session.delete(obj)
            await self.session.commit()
            return True
        except Exception as e:
            await self.session.rollback()
            logger.error("[delete] %s", e)
            return False

    # ---------------------------
    # Conteo + Paginado con búsqueda/joins/orden/filtros
    # ---------------------------
    async def count(self, search: Optional[str] = None, filters: Optional[Dict[str, Any]] = None,
                    joins: Optional[List[str]] = None, include_joined_in_search: bool = True) -> int:
        joins_needed = set(joins or [])
        search_expr = self._build_search(search, joins_needed, joins, include_joined_in_search)
        where_parts = []
        if search_expr is not None:
            where_parts.append(search_expr)
        where_parts.extend(self._build_filters(filters, joins_needed))
        stmt = select(func.count()).select_from(self.model)
        stmt = self._apply_joins(stmt, joins_needed)
        if where_parts:
            stmt = stmt.where(and_(*where_parts))
        res = await self.session.execute(stmt)
        return int(res.scalar() or 0)

    async def list_paginated(self, page: int = 1, per_page: int = 20,
                             search: Optional[str] = None, filters: Optional[Dict[str, Any]] = None,
                             order_by: Optional[str] = None, order_dir: str = "asc",
                             joins: Optional[List[str]] = None, include_joined_in_search: bool = True) -> Dict[str, Any]:
        page = max(1, int(page)); per_page = max(1, int(per_page))
        offset = (page - 1) * per_page
        joins_needed = set(joins or [])
        search_expr = self._build_search(search, joins_needed, joins, include_joined_in_search)
        where_parts = []
        if search_expr is not None:
            where_parts.append(search_expr)
        where_parts.extend(self._build_filters(filters, joins_needed))
        order_clause = self._build_order(order_by, order_dir, joins_needed)
        stmt = select(self.model)
        stmt = self._apply_joins(stmt, joins_needed)
        if where_parts:
            stmt = stmt.where(and_(*where_parts))
        if order_clause is not None:
            stmt = stmt.order_by(order_clause)
        stmt = stmt.offset(offset).limit(per_page)
        res = await self.session.execute(stmt)
        items = res.scalars().all()
        total = await self.count(search=search, filters=filters, joins=list(joins_needed),
                                 include_joined_in_search=include_joined_in_search)
        return {"items": items, "total": total}

    # ---------------------------
    # Helpers por relación (joinedload)
    # ---------------------------
    async def get_with(self, entity_id: Any, relations: List[str]) -> Optional[T]:
        stmt = select(self.model)
        for rel in relations:
            stmt = stmt.options(joinedload(getattr(self.model, rel)))
        stmt = stmt.where(getattr(self.model, "id") == entity_id) if "id" in self._columns_map() else stmt
        res = await self.session.execute(stmt)
        return res.scalars().first()
"""
REPO_TEMPLATE = """\
import logging
from typing import Optional, List, Dict, Any
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from {models_pkg} import {class_name}{junction_import}
from {repos_base_pkg} import AsyncBaseRepository

logger = logging.getLogger(__name__)

DEFAULT_RELATIONS = [{default_relations}]

class {class_name}Repository(AsyncBaseRepository[{class_name}]):
    \"\"\"Repositorio para {class_name}. (Auto-generado nivel 10)\"\"\"

    def __init__(self, session: AsyncSession):
        super().__init__({class_name}, session)

    # CRUD base está en AsyncBaseRepository (create, update, delete, etc.)

{delete_method}
{empresa_methods}
{text_search_helpers}
{relation_helpers}
{m2m_methods}
"""
SOFT_DELETE_METHOD = """\
    async def delete_entity(self, entity_id: int) -> bool:
        try:
            obj = await self.get_by_id(entity_id)
            if not obj:
                logger.warning(f"[{class_name}Repository.delete_entity] No se encontró ID {{entity_id}}")
                return False
            setattr(obj, "{soft_field}", "{soft_yes}")
            await self.session.commit()
            return True
        except Exception as e:
            await self.session.rollback()
            logger.error(f"[{class_name}Repository.delete_entity] Error: {{e}}")
            return False
"""
HARD_DELETE_METHOD = """\
    async def delete_entity(self, entity_id: int) -> bool:
        try:
            obj = await self.get_by_id(entity_id)
            if not obj:
                logger.warning(f"[{class_name}Repository.delete_entity] No se encontró ID {{entity_id}}")
                return False
            self.session.delete(obj)
            await self.session.commit()
            return True
        except Exception as e:
            await self.session.rollback()
            logger.error(f"[{class_name}Repository.delete_entity] Error: {{e}}")
            return False
"""
EMPRESA_METHODS = """\
    async def get_by_empresa(self, id_empresa: int) -> List[{class_name}]:
        try:
            stmt = select(self.model).where(
                self.model.id_empresa == id_empresa{not_deleted_clause}
            )
            res = await self.session.execute(stmt)
            return res.scalars().all()
        except Exception as e:
            logger.error(f"[{class_name}Repository.get_by_empresa] Error: {{e}}")
            return []

    async def count_all_by_empresa(self, id_empresa: int) -> int:
        try:
            stmt = select(self.model).where(
                self.model.id_empresa == id_empresa{not_deleted_clause}
            )
            res = await self.session.execute(stmt)
            return len(res.scalars().all())
        except Exception as e:
            logger.error(f"[{class_name}Repository.count_all_by_empresa] Error: {{e}}")
            return 0

    async def count_filtered_by_empresa(self, id_empresa: int, search: str) -> int:
        try:
            stmt = select(self.model).where(
                self.model.id_empresa == id_empresa{not_deleted_clause},
                or_(
{or_lines_emp}
                )
            )
            res = await self.session.execute(stmt)
            return len(res.scalars().all())
        except Exception as e:
            logger.error(f"[{class_name}Repository.count_filtered_by_empresa] Error: {{e}}")
            return 0

    async def get_paginated_by_empresa(self, id_empresa: int, start: int, length: int, search: str) -> List[{class_name}]:
        try:
            stmt = (
                select(self.model)
                .where(
                    self.model.id_empresa == id_empresa{not_deleted_clause},
                    or_(
{or_lines_emp}
                    )
                )
                .offset(start)
                .limit(length)
            )
            res = await self.session.execute(stmt)
            return res.scalars().all()
        except Exception as e:
            logger.error(f"[{class_name}Repository.get_paginated_by_empresa] Error: {{e}}")
            return []
"""
TEXT_SEARCH_HELPERS = """\
    async def count_filtered(self, search: str) -> int:
        try:
            stmt = select(self.model).where(
                or_(
{or_lines}
                )
            )
            res = await self.session.execute(stmt)
            return len(res.scalars().all())
        except Exception as e:
            logger.error(f"[{class_name}Repository.count_filtered] Error: {{e}}")
            return 0

    async def get_paginated(self, start: int, length: int, search: str) -> List[{class_name}]:
        try:
            stmt = (
                select(self.model)
                .where(
                    or_(
{or_lines}
                    )
                )
                .offset(start)
                .limit(length)
            )
            res = await self.session.execute(stmt)
            return res.scalars().all()
        except Exception as e:
            logger.error(f"[{class_name}Repository.get_paginated] Error: {{e}}")
            return []
"""
REL_HELPERS_TEMPLATE = """\
    # ---------- Helpers por relación (auto) ----------
{helpers}
"""
REL_HELPER_BLOCK = """\
    async def get_with_{rel}(self, entity_id: int) -> Optional[{class_name}]:
        \"\"\"Devuelve {class_name} + joinedload('{rel}')\"\"\"
        try:
            stmt = (
                select(self.model)
                .options(joinedload(self.model.{rel}))
                .where(self.model.id == entity_id)
            )
            res = await self.session.execute(stmt)
            return res.scalars().first()
        except Exception as e:
            logger.error(f"[{class_name}Repository.get_with_{rel}] Error: {{e}}")
            return None

    async def list_with_{rel}(self, page: int = 1, per_page: int = 20, search: Optional[str] = None) -> Dict[str, Any]:
        \"\"\"Lista paginada uniendo '{rel}', buscando también en columnas de esa relación.\"\"\"
        try:
            return await self.list_paginated(page=page, per_page=per_page, search=search, joins=["{rel}"], include_joined_in_search=True)
        except Exception as e:
            logger.error(f"[{class_name}Repository.list_with_{rel}] Error: {{e}}")
            return {{"items": [], "total": 0}}
"""
REL_FK_HELPER = """\
    async def get_by_{rel}(self, {rel}_id: int) -> List[{class_name}]:
        \"\"\"Filtra por FK local '{rel}_id' si existe.\"\"\"
        try:
            stmt = select(self.model).where(getattr(self.model, "{rel}_id") == {rel}_id)
            res = await self.session.execute(stmt)
            return res.scalars().all()
        except Exception as e:
            logger.error(f"[{class_name}Repository.get_by_{rel}] Error: {{e}}")
            return []
"""
M2M_METHODS = """\
    async def agregar_relacion(self, {main_id_arg}: int, {other_id_arg}: int) -> bool:
        try:
            rel = {junction_class}({main_id_arg}={main_id_arg}, {other_id_arg}={other_id_arg}{junction_soft_init})
            self.session.add(rel)
            await self.session.commit()
            return True
        except Exception as e:
            await self.session.rollback()
            logger.error(f"[{class_name}Repository.agregar_relacion] Error: {{e}}")
            return False

    async def eliminar_relacion(self, {main_id_arg}: int, {other_id_arg}: int) -> bool:
        try:
            stmt = select({junction_class}).where(
                {junction_class}.{main_id_arg} == {main_id_arg},
                {junction_class}.{other_id_arg} == {other_id_arg}
            )
            res = await self.session.execute(stmt)
            rel = res.scalars().first()
            if not rel:
                logger.warning(f"[{class_name}Repository.eliminar_relacion] Relación no encontrada")
                return False
            # Hard/soft
            if hasattr(rel, "borrado"):
                setattr(rel, "borrado", "si")
            else:
                await self.session.delete(rel)
            await self.session.commit()
            return True
        except Exception as e:
            await self.session.rollback()
            logger.error(f"[{class_name}Repository.eliminar_relacion] Error: {{e}}")
            return False
"""
def _column_names(cls) -> Iterable[str]:
    try:
        return [c.name for c in cls.__table__.columns]  # type: ignore[attr-defined]
    except Exception:
        return []
def _text_columns(cls) -> List[str]:
    out = []
    try:
        for c in cls.__table__.columns:  # type: ignore[attr-defined]
            tname = c.type.__class__.__name__.lower()
            if (("string" in tname) or ("text" in tname) or ("varchar" in tname) or ("citext" in tname)) and not re.search(SENSITIVE_RE, c.name, re.I):
                out.append(c.name)
    except Exception:
        pass
    return out
def _relationship_names(cls) -> List[str]:
    try:
        return [r.key for r in cls.__mapper__.relationships]  # type: ignore[attr-defined]
    except Exception:
        return []
def repositorios_base_pkg_sane(pkg: str) -> str:
    return pkg.split(":")[0]
@app.command(name="generate-repositories", help="Genera repos avanzados (un archivo por modelo) con joins y helpers por relación")
def generate_repositories(
    models_pkg: str = "modelos",
    out_dir: str = "repositorios",
    repos_base_pkg: str = "repositorios.base",
    soft_delete_field: str = "borrado",
    soft_delete_yes_value: str = "no",
):
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    base_repo_file = out_path / "base.py"
    if not base_repo_file.exists():
        base_repo_file.write_text(REPOS_BASE_FILE, encoding="utf-8")
        typer.secho(f"✓ {base_repo_file} creado", fg=typer.colors.GREEN)
    modelos = importlib.import_module(models_pkg)
    Base = getattr(modelos, "Base")
    modelos_clases = []
    for name, obj in inspect.getmembers(modelos):
        if inspect.isclass(obj) and obj is not Base and issubclass(obj, Base):
            modelos_clases.append(obj)
    if not modelos_clases:
        typer.secho("⚠ No encontré modelos en el paquete especificado.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    created = []
    for cls in modelos_clases:
        class_name = cls.__name__
        cols = set(_column_names(cls))
        txt_cols = _text_columns(cls)
        rels = _relationship_names(cls)
        has_soft = soft_delete_field in cols
        has_empresa = "id_empresa" in cols
        delete_code = (SOFT_DELETE_METHOD if has_soft else HARD_DELETE_METHOD).format(
            class_name=class_name, soft_field=soft_delete_field, soft_yes="si"
        )
        text_search_helpers = ""
        if txt_cols:
            or_lines = ",\n".join([f"                    self.model.{c}.ilike(f\"%{{search}}%\")" for c in txt_cols])
            text_search_helpers = TEXT_SEARCH_HELPERS.format(class_name=class_name, or_lines=or_lines)
        empresa_methods = ""
        if has_empresa:
            not_deleted_clause = f", self.model.{soft_delete_field} == '{soft_delete_yes_value}'" if has_soft else ""
            or_lines_emp = ",\n".join([f"                    self.model.{c}.ilike(f\"%{{search}}%\")" for c in (txt_cols or ["id_empresa"])])
            empresa_methods = EMPRESA_METHODS.format(
                class_name=class_name,
                not_deleted_clause=not_deleted_clause,
                or_lines_emp=or_lines_emp,
            )
        helpers_blocks = []
        for rel in rels:
            helpers_blocks.append(REL_HELPER_BLOCK.format(class_name=class_name, rel=rel))
            if f"{rel}_id" in cols:
                helpers_blocks.append(REL_FK_HELPER.format(class_name=class_name, rel=rel))
        relation_helpers = REL_HELPERS_TEMPLATE.format(helpers="\n".join(helpers_blocks)) if helpers_blocks else ""
        junction_import = ""
        m2m_methods = ""
        main_lower = class_name.lower()
        junction_candidate = None
        for name2, obj2 in inspect.getmembers(modelos):
            if inspect.isclass(obj2) and obj2 is not Base and issubclass(obj2, Base):
                cols2 = set(_column_names(obj2))
                if f"{main_lower}_id" in cols2:
                    other_ids = [c for c in cols2 if c.endswith("_id") and c != f"{main_lower}_id"]
                    if other_ids:
                        junction_candidate = (obj2.__name__, other_ids[0], cols2)
                        break
        if junction_candidate:
            junction_class, other_id, j_cols = junction_candidate
            junction_import = f", {junction_class}"
            junction_soft_init = f", {soft_delete_field}='{soft_delete_yes_value}'" if soft_delete_field in j_cols else ""
            m2m_methods = M2M_METHODS.format(
                class_name=class_name,
                junction_class=junction_class,
                main_id_arg=f"{main_lower}_id",
                other_id_arg=other_id,
                junction_soft_init=junction_soft_init,
            )

        default_relations = ", ".join([f"'{r}'" for r in rels])  # por si querés usarlas fácil

        code = REPO_TEMPLATE.format(
            class_name=class_name,
            models_pkg=models_pkg,
            repos_base_pkg=repositorios_base_pkg_sane(repos_base_pkg),
            delete_method=delete_code,
            empresa_methods=empresa_methods,
            text_search_helpers=text_search_helpers,
            relation_helpers=relation_helpers,
            m2m_methods=m2m_methods,
            junction_import=junction_import,
            default_relations=default_relations
        )

        repo_filename = f"{class_name.lower()}_repository.py"
        repo_path = out_path / repo_filename
        if repo_path.exists():
            typer.secho(f"• {repo_path} ya existe, lo omito.", fg=typer.colors.YELLOW)
        else:
            repo_path.write_text(code, encoding="utf-8")
            created.append((class_name, repo_filename))
            typer.secho(f"✓ {repo_path} generado", fg=typer.colors.GREEN)

    # __init__.py de repos
    init_lines = ["# Auto-generado. Exporta repositorios.\n"]
    for class_name, repo_filename in created:
        mod = repo_filename[:-3]
        init_lines.append(f"from .{mod} import {class_name}Repository\n")
    init_lines.append("\n__all__ = [\n")
    for class_name, _ in created:
        init_lines.append(f"    '{class_name}Repository',\n")
    init_lines.append("]\n")
    (out_path / "__init__.py").write_text("".join(init_lines), encoding="utf-8")

    typer.secho(f"Listo ✅ Repos generados en {out_path}", fg=typer.colors.GREEN)

SERVICE_TEMPLATE = """\
import logging
from typing import Optional, List, Dict, Any

from {models_pkg} import {class_name}
from {repos_pkg} import {class_name}Repository

logger = logging.getLogger(__name__)

class {class_name}Service:
    \"\"\"Servicio para {class_name}. Auto-generado.\"\"\"

    def __init__(self, repo: {class_name}Repository):
        self.repo = repo

    async def crear(self, data: dict) -> Optional[{class_name}]:
        try:
            return await self.repo.create(data)
        except Exception as e:
            logger.error(f"[{class_name}Service.crear] Error: {{e}}")
            return None

    async def actualizar(self, id_: int, data: dict) -> Optional[{class_name}]:
        try:
            obj = await self.repo.get_by_id(id_)
            if not obj:
                logger.warning(f"[{class_name}Service.actualizar] No encontrado: {{id_}}")
                return None
            return await self.repo.update(obj, data)
        except Exception as e:
            logger.error(f"[{class_name}Service.actualizar] Error: {{e}}")
            return None

    async def eliminar(self, id_: int) -> bool:
        try:
            return await self.repo.delete(id_)
        except Exception as e:
            logger.error(f"[{class_name}Service.eliminar] Error: {{e}}")
            return False

    async def obtener_todas(self) -> List[{class_name}]:
        try:
            return await self.repo.get_all()
        except Exception as e:
            logger.error(f"[{class_name}Service.obtener_todas] Error: {{e}}")
            return []

    async def obtener_por_id(self, id_: int) -> Optional[{class_name}]:
        try:
            return await self.repo.get_by_id(id_)
        except Exception as e:
            logger.error(f"[{class_name}Service.obtener_por_id] Error: {{e}}")
            return None
{slug_method}
{text_methods}
{empresa_methods}
{relation_methods}
"""
SERVICE_EMPRESA_METHODS = """\
    # --- Alcance por empresa ---
    async def obtener_por_empresa(self, id_empresa: int) -> List[{class_name}]:
        try:
            return await self.repo.get_by_empresa(id_empresa)
        except Exception as e:
            logger.error(f"[{class_name}Service.obtener_por_empresa] Error: {{e}}")
            return []

    async def count_all_by_empresa(self, id_empresa: int) -> int:
        try:
            return await self.repo.count_all_by_empresa(id_empresa)
        except Exception as e:
            logger.error(f"[{class_name}Service.count_all_by_empresa] Error: {{e}}")
            return 0

    async def count_filtered_by_empresa(self, id_empresa: int, search: str) -> int:
        try:
            return await self.repo.count_filtered_by_empresa(id_empresa, search)
        except Exception as e:
            logger.error(f"[{class_name}Service.count_filtered_by_empresa] Error: {{e}}")
            return 0

    async def get_paginated_by_empresa(self, id_empresa: int, start: int, length: int, search: str) -> List[{class_name}]:
        try:
            return await self.repo.get_paginated_by_empresa(id_empresa, start, length, search)
        except Exception as e:
            logger.error(f"[{class_name}Service.get_paginated_by_empresa] Error: {{e}}")
            return []
"""
SERVICE_TEXT_METHODS = """\
    # --- Búsqueda/paginación por texto ---
    async def count_filtered(self, search: str) -> int:
        try:
            return await self.repo.count_filtered(search)
        except Exception as e:
            logger.error(f"[{class_name}Service.count_filtered] Error: {{e}}")
            return 0

    async def get_paginated(self, start: int, length: int, search: str) -> List[{class_name}]:
        try:
            return await self.repo.get_paginated(start, length, search)
        except Exception as e:
            logger.error(f"[{class_name}Service.get_paginated] Error: {{e}}")
            return []
"""
SERVICE_SLUG_METHOD = """\
    # --- Helpers específicos ---
    async def get_by_slug(self, slug: str) -> Optional[{class_name}]:
        try:
            return await self.repo.get_by_property("slug", slug)
        except Exception as e:
            logger.error(f"[{class_name}Service.get_by_slug] Error: {{e}}")
            return None
"""
SERVICE_RELATION_BLOCK = """\
    # --- Helpers por relación ---
    async def get_with_{rel}(self, entity_id: int) -> Optional[{class_name}]:
        try:
            return await self.repo.get_with_{rel}(entity_id)
        except Exception as e:
            logger.error(f"[{class_name}Service.get_with_{rel}] Error: {{e}}")
            return None

    async def list_with_{rel}(self, page: int = 1, per_page: int = 20, search: str | None = None) -> Dict[str, Any]:
        try:
            return await self.repo.list_with_{rel}(page=page, per_page=per_page, search=search)
        except Exception as e:
            logger.error(f"[{class_name}Service.list_with_{rel}] Error: {{e}}")
            return {{"items": [], "total": 0}}
"""
def _column_names(cls) -> Iterable[str]:
    try:
        return [c.name for c in cls.__table__.columns]  # type: ignore[attr-defined]
    except Exception:
        return []
def _text_columns(cls) -> List[str]:
    out = []
    try:
        for c in cls.__table__.columns:  # type: ignore[attr-defined]
            tname = c.type.__class__.__name__.lower()
            if (("string" in tname) or ("text" in tname) or ("varchar" in tname) or ("citext" in tname)) and not re.search(SENSITIVE_RE, c.name, re.I):
                out.append(c.name)
    except Exception:
        pass
    return out
def _relationship_names(cls) -> List[str]:
    try:
        return [r.key for r in cls.__mapper__.relationships]  # type: ignore[attr-defined]
    except Exception:
        return []
@app.command(name="generate-services", help="Genera services por cada repo/modelo")
def generate_services(
    models_pkg: str = "modelos",
    repos_pkg: str = "repositorios",
    out_dir: str = "services",
):
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    modelos = importlib.import_module(models_pkg)
    repos = importlib.import_module(repos_pkg)
    Base = getattr(modelos, "Base")
    modelos_clases = []
    for name, obj in inspect.getmembers(modelos):
        if inspect.isclass(obj) and obj is not Base and issubclass(obj, Base):
            modelos_clases.append(obj)
    if not modelos_clases:
        typer.secho("⚠ No encontré modelos en el paquete especificado.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    created = []
    for cls in modelos_clases:
        class_name = cls.__name__
        cols = set(_column_names(cls))
        txt_cols = _text_columns(cls)
        rels = _relationship_names(cls)
        slug_method = SERVICE_SLUG_METHOD.format(class_name=class_name) if "slug" in cols else ""
        text_methods = SERVICE_TEXT_METHODS.format(class_name=class_name) if txt_cols else ""
        empresa_methods = ""
        if "id_empresa" in cols:
            empresa_methods = SERVICE_EMPRESA_METHODS.format(class_name=class_name)
        relation_methods = ""
        if rels:
            blocks = [SERVICE_RELATION_BLOCK.format(class_name=class_name, rel=r) for r in rels]
            relation_methods = "\n".join(blocks)
        code = SERVICE_TEMPLATE.format(
            class_name=class_name,
            models_pkg=models_pkg,
            repos_pkg=repos_pkg + '.' + f"{class_name.lower()}_repository",
            slug_method=slug_method,
            text_methods=text_methods,
            empresa_methods=empresa_methods,
            relation_methods=relation_methods,
        )
        svc_filename = f"{class_name.lower()}_service.py"
        svc_path = out_path / svc_filename
        if svc_path.exists():
            typer.secho(f"• {svc_path} ya existe, lo omito.", fg=typer.colors.YELLOW)
        else:
            svc_path.write_text(code, encoding="utf-8")
            created.append((class_name, svc_filename))
            typer.secho(f"✓ {svc_path} generado", fg=typer.colors.GREEN)
    init_lines = ["# Auto-generado. Exporta services.\n"]
    for class_name, svc_filename in created:
        mod = svc_filename[:-3]
        init_lines.append(f"from .{mod}.{class_name.lower()} import {class_name}Service\n")
    init_lines.append("\n__all__ = [\n")
    for class_name, _ in created:
        init_lines.append(f"    '{class_name}Service',\n")
    init_lines.append("]\n")
    (out_path / "__init__.py").write_text("".join(init_lines), encoding="utf-8")
    typer.secho(f"Listo ✅ Services generados en {out_path}", fg=typer.colors.GREEN)
SCHEMAS_SENSITIVE_RE = re.compile(r"(pass|password|clave|secret|token|salt|hash)", re.IGNORECASE)
def _sa_columns_for_schema(cls):
    try:
        return list(cls.__table__.columns)  # type: ignore[attr-defined]
    except Exception:
        return []
def _sa_pk_names_for_schema(cls) -> List[str]:
    try:
        return [c.name for c in cls.__table__.primary_key]  # type: ignore[attr-defined]
    except Exception:
        return []
def _is_sensitive_col_schema(name: str) -> bool:
    return bool(SCHEMAS_SENSITIVE_RE.search(name))
def _py_type_for_sa_type(col) -> Any:
    t = col.type
    tname = t.__class__.__name__.lower()
    if "int" in tname or "integer" in tname or "smallinteger" in tname or "biginteger" in tname:
        return int
    if "float" in tname:
        return float
    if "numeric" in tname or "decimal" in tname:
        return decimal.Decimal
    if "bool" in tname:
        return bool
    if "uuid" in tname:
        return uuid.UUID
    if "date" == tname:
        return datetime.date
    if "datetime" in tname:
        return datetime.datetime
    if "time" == tname:
        return datetime.time
    if "largebinary" in tname or "binary" in tname or "bytea" in tname:
        return bytes
    if "json" in tname:
        return Any
    if "string" in tname or "varchar" in tname or "text" in tname or "citext" in tname:
        return str
    if "double_precision" in tname:
        return float
    if "array" in tname:
        try:
            item_t = getattr(t, "item_type", None)
            if item_t is not None:
                fake_col = type("C", (), {"type": item_t})()
                inner = _py_type_for_sa_type(fake_col)
                from typing import List as _List  # evitar shadowing
                return _List[inner]  # type: ignore
        except Exception:
            pass
        from typing import List as _List
        return _List[Any]
    return Any
def _annotation_schema(col, *, optional: bool) -> str:
    py_t = _py_type_for_sa_type(col)
    if py_t is Any:
        ann = "Any"
    elif py_t is int:
        ann = "int"
    elif py_t is float:
        ann = "float"
    elif py_t is bool:
        ann = "bool"
    elif py_t is str:
        ann = "str"
    elif py_t is bytes:
        ann = "bytes"
    elif py_t is decimal.Decimal:
        ann = "decimal.Decimal"
    elif py_t is datetime.date:
        ann = "datetime.date"
    elif py_t is datetime.datetime:
        ann = "datetime.datetime"
    elif py_t is datetime.time:
        ann = "datetime.time"
    elif py_t is uuid.UUID:
        ann = "uuid.UUID"
    else:
        ann = str(py_t).replace("typing.", "")
    return f"Optional[{ann}]" if optional else ann
SCHEMA_HEADER = """\
from __future__ import annotations
from typing import Any, Optional, List
import uuid
import decimal
import datetime
from pydantic import BaseModel, Field
"""
SCHEMA_TEMPLATE = """\
{header}

class {class_name}Base(BaseModel):
{base_fields}
    model_config = {{"from_attributes": True}}

class {class_name}Create({class_name}Base):
{create_fields}
    pass

class {class_name}Update(BaseModel):
{update_fields}
    model_config = {{"from_attributes": True}}

class {class_name}Out(BaseModel):
{out_fields}
    model_config = {{"from_attributes": True}}
"""
def _indent_schema(lines: List[str], spaces: int = 4) -> str:
    pad = " " * spaces
    return "".join(pad + ln for ln in lines)
def _default_for_col_schema(col):
    if getattr(col, "default", None) is not None or getattr(col, "server_default", None) is not None:
        return None, True
    return ..., not col.nullable
def _build_schema_code_for_model(cls) -> str:
    cols = _sa_columns_for_schema(cls)
    if not cols:
        return ""

    pk_names = set(_sa_pk_names_for_schema(cls))

    base_lines: List[str] = []
    for col in cols:
        if col.name in pk_names:
            continue
        optional = col.nullable or getattr(col, "default", None) is not None or getattr(col, "server_default", None) is not None
        ann = _annotation_schema(col, optional=optional)
        default, required = _default_for_col_schema(col)
        if default is ... and optional:
            default = None
        if default is ...:
            base_lines.append(f"{col.name}: {ann}\n")
        else:
            base_lines.append(f"{col.name}: {ann} = Field(default={repr(default)})\n")

    create_lines: List[str] = []  # hereda de Base

    update_lines: List[str] = []
    for col in cols:
        if col.name in pk_names:
            continue
        ann_opt = _annotation_schema(col, optional=True)
        update_lines.append(f"{col.name}: {ann_opt} = None\n")

    out_lines: List[str] = []
    for col in cols:
        if _is_sensitive_col_schema(col.name):
            continue
        optional = col.nullable
        ann = _annotation_schema(col, optional=optional)
        out_lines.append(f"{col.name}: {ann}\n")

    code = SCHEMA_TEMPLATE.format(
        header=SCHEMA_HEADER,
        class_name=cls.__name__,
        base_fields=_indent_schema(base_lines) if base_lines else _indent_schema(["pass\n"]),
        create_fields=_indent_schema(create_lines) if create_lines else _indent_schema(["pass\n"]),
        update_fields=_indent_schema(update_lines) if update_lines else _indent_schema(["pass\n"]),
        out_fields=_indent_schema(out_lines) if out_lines else _indent_schema(["pass\n"]),
    )
    return code
@app.command(name="generate-schemas", help="Genera Pydantic schemas (Base/Create/Update/Out) por modelo")
def generate_schemas(models_pkg: str = "modelos", out_dir: str = "schemas"):
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    modelos = importlib.import_module(models_pkg)
    Base = getattr(modelos, "Base")
    modelos_clases = []
    for name, obj in inspect.getmembers(modelos):
        if inspect.isclass(obj) and obj is not Base and issubclass(obj, Base):
            modelos_clases.append(obj)
    if not modelos_clases:
        typer.secho("⚠ No encontré modelos en el paquete especificado.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    created = []
    for cls in modelos_clases:
        schema_filename = f"{cls.__name__.lower()}_schemas.py"
        schema_path = out_path / schema_filename
        if schema_path.exists():
            typer.secho(f"• {schema_path} ya existe, lo omito.", fg=typer.colors.YELLOW)
            continue
        code = _build_schema_code_for_model(cls)
        if not code:
            typer.secho(f"• {cls.__name__}: sin columnas, omito.", fg=typer.colors.YELLOW)
            continue
        schema_path.write_text(code, encoding="utf-8")
        created.append((cls.__name__, schema_filename))
        typer.secho(f"✓ {schema_path} generado", fg=typer.colors.GREEN)
    init_lines = ["# Auto-generado. Exporta schemas.\n"]
    for class_name, schema_filename in created:
        mod = schema_filename[:-3]
        init_lines.append(f"from .{mod} import {class_name}Base, {class_name}Create, {class_name}Update, {class_name}Out\n")
    init_lines.append("\n__all__ = [\n")
    for class_name, _ in created:
        init_lines.append(f"    '{class_name}Base', '{class_name}Create', '{class_name}Update', '{class_name}Out',\n")
    init_lines.append("]\n")
    (out_path / "__init__.py").write_text("".join(init_lines), encoding="utf-8")
    typer.secho(f"Listo ✅ Schemas generados en {out_path}", fg=typer.colors.GREEN)
CTRL_HEADER = """\
from fastapi import APIRouter, Depends, Query, Path, HTTPException, Request
from typing import List, Dict, Any, Optional
from {services_pkg} import {service_class}

# --- dependencias opcionales (si tu proyecto las tiene) ---
try:
    from dependencies.auth import get_current_user
except Exception:
    async def get_current_user():
        # stub muy simple; reemplazá por tu implementación real
        return type("U", (), {{}})()

# TODO: reemplazar por tu inyección real
def get_service() -> {service_class}:
    # Ejemplo: wirear con tu contenedor/Session en la vida real
    return {service_class}(repo=None)  # placeholder

router = APIRouter(prefix="/{resource}", tags=["{role}:{resource}"])
"""
CTRL_AJAX = """\
@router.post("/ajax", response_model=dict)
async def {resource}_ajax(
    request: Request,
    svc: {service_class} = Depends(get_service),
):
    try:
        form = await request.form()
        draw = int(form.get("draw", 1))
        start = int(form.get("start", 0))
        length = int(form.get("length", 10))
        search_value = form.get("search[value]", "")

        columns = {columns_json}

        # Si el service tiene helpers de texto, úsalos; si no, fallback simple
        try:
            total = await svc.count_filtered(search_value)
            page_items = await svc.get_paginated(start, length, search_value)
        except Exception:
            all_items = await svc.obtener_todas()
            total = len(all_items)
            page_items = all_items[start:start+length]

        data = []
        for obj in page_items or []:
            row = {{
                {row_kv_pairs},
                "acciones": getattr(obj, "{pk}", None)
            }}
            data.append(row)

        totalPages = (total + length - 1)//length if length>0 else 1
        return {{
            "draw": draw,
            "columns": columns,
            "iTotalRecords": total,
            "iTotalDisplayRecords": total,
            "data": data,
            "totalPages": totalPages
        }}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error al cargar {resource}")
"""
CTRL_CREATE = """\
@router.post("/", response_model=Dict[str, Any])
async def create_(payload: Dict[str, Any], svc: {service_class} = Depends(get_service)):
    obj = await svc.crear(payload)
    if not obj:
        raise HTTPException(status_code=400, detail="No se pudo crear")
    return {{"ok": True, "{pk}": getattr(obj, "{pk}", None)}}
"""
CTRL_LIST = """\
@router.get("/", response_model=Dict[str, Any])
async def list_(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=200),
    search: Optional[str] = Query(None),
    id_empresa: Optional[int] = Query(None),
    svc: {service_class} = Depends(get_service),
    user: Any = Depends(get_current_user),
):
    try:
        # Deducir id_empresa: prioridad a query, luego user.id_empresa
        id_emp = id_empresa
        if id_emp is None:
            id_emp = getattr(user, "id_empresa", None)

        # Si el service soporta scope por empresa, usarlo
        if id_emp is not None and all(hasattr(svc, m) for m in ("get_paginated_by_empresa", "count_filtered_by_empresa")):
            start = (page - 1) * per_page
            items = await svc.get_paginated_by_empresa(id_emp, start, per_page, search or "")
            total = await svc.count_filtered_by_empresa(id_emp, search or "")
            return {{"items": items, "total": total, "page": page, "per_page": per_page}}

        # Fallback: usar paginado genérico del repo
        try:
            res = await svc.repo.list_paginated(page=page, per_page=per_page, search=search)  # type: ignore
            res.setdefault("page", page); res.setdefault("per_page", per_page)
            return res
        except Exception:
            items = await svc.obtener_todas()
            return {{"items": items, "total": len(items), "page": page, "per_page": per_page}}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al listar: {{e}}")
"""
CTRL_UPDATE = """\
@router.put("/{{{param_name}}}", response_model=Dict[str, Any])
async def update_({param_name}: int = Path(..., ge=1), payload: Dict[str, Any] | None = None, svc: {service_class} = Depends(get_service)):
    try:
        obj = await svc.actualizar({param_name}, payload or {{}})
        if not obj:
            raise HTTPException(status_code=404, detail="No encontrado")
        return {{"ok": True}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al actualizar: {{e}}")
"""
CTRL_DELETE = """\
@router.delete("/{{{param_name}}}", response_model=Dict[str, Any])
async def delete_({param_name}: int = Path(..., ge=1), svc: {service_class} = Depends(get_service)):
    try:
        ok = await svc.eliminar({param_name})
        if not ok:
            raise HTTPException(status_code=404, detail="No encontrado")
        return {{"ok": True}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al eliminar: {{e}}")
"""
CTRL_VIEW = """\
@router.get("/{{{param_name}}}", response_model=Dict[str, Any])
async def get_one({param_name}: int = Path(..., ge=1), svc: {service_class} = Depends(get_service)):
    try:
        obj = await svc.obtener_por_id({param_name})
        if not obj:
            raise HTTPException(status_code=404, detail="No encontrado")
        return {{"item": obj}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener: {{e}}")
"""
@app.command(name="generate-controllers", help="Genera controladores FastAPI por rol/módulo desde roles_controlador.yaml (auto PK/columnas)")
def generate_controllers(
    services_pkg: str = "services",
    roles_yaml: str = "roles_controlador.yaml",
    out_dir: str = "controladores",
    models_pkg: str = "modelos",
):
    out_path = _pl.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if not _pl.Path(roles_yaml).exists():
        typer.secho(f"⚠ No encontré {roles_yaml}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    with open(roles_yaml, "r", encoding="utf-8") as f:
        roles = yaml.safe_load(f) or {}
    created = 0
    for role, recursos in (roles or {}).items():
        role_dir = out_path / role
        role_dir.mkdir(parents=True, exist_ok=True)
        for recurso, perms in (recursos or {}).items():
            service_class = service_class_for(recurso, role, services_pkg)
            model_cls = _find_model_class(models_pkg, recurso)
            pk = _ctrl_pk_name(model_cls) if model_cls else "id"
            show_cols = _ctrl_string_cols(model_cls) if model_cls else []
            if not show_cols and model_cls:
                show_cols = _ctrl_non_sensitive(model_cls)[:4]
            if not show_cols:
                show_cols = ["nombre", "descripcion", "email", "telefono"]  # fallback
            columns_json = json.dumps(
                [{"field": c, "label": c.capitalize()} for c in show_cols] + [
                    {"field": "acciones", "label": "Acciones"}],
                ensure_ascii=False
            )
            row_kv_pairs = ", ".join([f'"{c}": getattr(obj, "{c}", None)' for c in show_cols])
            file_path = role_dir / f"{recurso}_controller.py"
            code = CTRL_HEADER.format(
                services_pkg=services_pkg,
                service_class=service_class,
                resource=recurso,
                role=role,
            )
            singular = _singular(recurso)
            param_name = PARAM_OVERRIDES.get(recurso, f"id_{singular}")
            if perms.get("list"):
                if "CTRL_AJAX" in globals():  # si tenés la plantilla AJAX
                    code += "\n" + CTRL_AJAX.format(
                        service_class=service_class,
                        resource=recurso,
                        columns_json=columns_json,
                        row_kv_pairs=row_kv_pairs,
                        pk=pk,
                    )
                code += "\n" + CTRL_LIST.format(service_class=service_class)
            if perms.get("create"):
                code += "\n" + CTRL_CREATE.format(service_class=service_class, pk=pk)
            if perms.get("update"):
                code += "\n" + CTRL_UPDATE.format(service_class=service_class, param_name=param_name)
            if perms.get("delete"):
                code += "\n" + CTRL_DELETE.format(service_class=service_class, param_name=param_name)
            if perms.get("view") or perms.get("ver"):
                code += "\n" + CTRL_VIEW.format(service_class=service_class, param_name=param_name)
            if code.strip().endswith("]"):
                typer.secho(f"• {role}/{recurso}: sin acciones, omito.", fg=typer.colors.YELLOW)
                continue
            file_path.write_text(code, encoding="utf-8")
            created += 1
            typer.secho(f"✓ {file_path} generado", fg=typer.colors.GREEN)
        init_lines = []
        for py in role_dir.glob("*_controller.py"):
            mod = py.stem
            init_lines.append(f"from .{mod} import router as {mod}_router\n")
        init_lines.append("\n__all__ = [\n")
        for py in role_dir.glob("*_controller.py"):
            init_lines.append(f"    '{py.stem}_router',\n")
        init_lines.append("]\n")
        (role_dir / "__init__.py").write_text("".join(init_lines), encoding="utf-8")
    root_init = []
    for role in roles.keys():
        root_init.append(f"from .{role} import *  # noqa\n")
    (out_path / "__init__.py").write_text("".join(root_init), encoding="utf-8")
    typer.secho(f"Listo ✅ Controladores generados en {out_path} ({created} archivos).", fg=typer.colors.GREEN)
def _snake_from_camel(name: str) -> str:
    s1 = _re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return _re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
def _discover_classes(module_name: str, suffix: str) -> Dict[str, type]:
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return {}
    out = {}
    for n, obj in inspect.getmembers(mod, inspect.isclass):
        if n.endswith(suffix):
            out[n] = obj
    return out
def _ensure_dependencies_pkg_dynamic():
    """
    Genera dependencies/dependencies.py dinámicamente a partir de los *Service y *Repository existentes.
    Empareja FooService -> FooRepository si existe.
    Además, siempre intenta exponer get_auth_service si están disponibles AuthService/AuthRepository.
    """
    dep_dir = _pl.Path("dependencies")
    dep_dir.mkdir(parents=True, exist_ok=True)
    (dep_dir / "__init__.py").write_text("", encoding="utf-8")
    auth_path = dep_dir / "auth.py"
    if not auth_path.exists():
        auth_path.write_text(
            "from fastapi import Depends, HTTPException, status\n"
            "from fastapi.security import OAuth2PasswordBearer\n"
            "from jose import jwt, JWTError\n\n"
            "from core.config import JWT_SECRET_KEY, ALGORITHM\n"
            "from modelos.usuarios import Usuarios\n"
            "from repositorios.usuarios_repository import UsuariosRepository\n"
            "from database import get_async_session\n\n"
            "oauth2_scheme = OAuth2PasswordBearer(tokenUrl=\"/login\")\n\n"
            "async def get_current_user(token: str = Depends(oauth2_scheme), session=Depends(get_async_session)) -> Usuarios:\n"
            "    credentials_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=\"No autorizado\")\n"
            "    try:\n"
            "        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])\n"
            "        username = payload.get(\"username\")\n"
            "        if username is None:\n"
            "            raise credentials_exception\n"
            "    except JWTError:\n"
            "        raise credentials_exception\n"
            "    repo = UsuariosRepository(session)\n"
            "    # Ajustá esto a tu método real de búsqueda (por username)\n"
            "    user = await repo.get_by_property('username', username) if hasattr(repo, 'get_by_property') else None\n"
            "    if user is None:\n"
            "        raise credentials_exception\n"
            "    return user\n\n"
            "def require_role(required_roles: list[str]):\n"
            "    async def role_checker(user: Usuarios = Depends(get_current_user)):\n"
            "        if getattr(user, 'rol', None) not in required_roles:\n"
            "            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=\"No tiene permiso para acceder a esta ruta\")\n"
            "        return user\n"
            "    return role_checker\n",
            encoding="utf-8",
        )
def _ensure_database_py():
    db_path = _pl.Path("database.py")
    if db_path.exists():
        return  # respetar tu archivo actual
    code = '''\
from __future__ import annotations
import os
from typing import AsyncGenerator
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

load_dotenv()

# Permite URL completa o piezas por separado
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    server = os.getenv("DB_SERVER", "localhost")
    port   = os.getenv("DB_PORT", "5432")
    user   = os.getenv("DB_USER", "postgres")
    pwd    = os.getenv("DB_PASSWORD", "")
    name   = os.getenv("DB_NAME", "")
    # driver async por defecto: asyncpg
    DATABASE_URL = f"postgresql+asyncpg://{user}:{pwd}@{server}:{port}/{name}"

engine = create_async_engine(DATABASE_URL, echo=bool(int(os.getenv("SQL_ECHO", "0"))))
async_session: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine, expire_on_commit=False
)

async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()
'''
    db_path.write_text(code, encoding="utf-8")
def ensure_core_modules(base_dir: str = "."):
    """
    Crea la carpeta 'core' con los módulos config.py, file_upload.py y security.py
    usando el código base provisto.
    """
    core_path = Path(base_dir) / "core"
    core_path.mkdir(parents=True, exist_ok=True)
    (core_path / "__init__.py").write_text("", encoding="utf-8")
    (core_path / "config.py").write_text(dedent('''\
        import os
        from dotenv import load_dotenv

        load_dotenv()

        JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "clave-secreta")
        ALGORITHM = "HS256"
        ACCESS_TOKEN_EXPIRE_MINUTES = 60

        SMTP_SERVER = os.getenv("SMTP_SERVER")
        SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
        SMTP_USER = os.getenv("SMTP_USER")
        SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
    '''), encoding="utf-8")
    (core_path / "file_upload.py").write_text(dedent('''\
        import os
        import uuid
        import shutil
        from pathlib import Path
        from fastapi import UploadFile
        from PIL import Image, ImageOps

        UPLOAD_FOLDER = "uploads"
        Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

        def generate_uuid_filename(extension: str) -> str:
            return f"{uuid.uuid4().hex}.{extension}"

        async def save_logo_file(file: UploadFile) -> str:
            try:
                temp_path = os.path.join(UPLOAD_FOLDER, generate_uuid_filename("tmp"))
                with open(temp_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                image = Image.open(temp_path)
                if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
                    image = image.convert("RGBA")
                else:
                    image = image.convert("RGB")
                image = ImageOps.fit(image, (1000, 1000), method=Image.LANCZOS, centering=(0.5, 0.5))
                filename = generate_uuid_filename("webp")
                final_path = os.path.join(UPLOAD_FOLDER, filename)
                image.save(final_path, "WEBP", quality=100, lossless=True)
                os.remove(temp_path)
                return f"/{final_path}"
            except Exception as e:
                raise RuntimeError(f"Error al guardar logo: {e}")

        async def save_producto_file(file: UploadFile) -> str:
            try:
                temp_path = os.path.join(UPLOAD_FOLDER, generate_uuid_filename("tmp"))
                with open(temp_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                image = Image.open(temp_path)
                if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
                    image = image.convert("RGBA")
                else:
                    image = image.convert("RGB")
                image = ImageOps.fit(image, (1000, 1000), method=Image.LANCZOS, centering=(0.5, 0.5))
                filename = generate_uuid_filename("webp")
                final_path = os.path.join(UPLOAD_FOLDER, filename)
                image.save(final_path, "WEBP", quality=100, lossless=True)
                os.remove(temp_path)
                return f"/{final_path}"
            except Exception as e:
                raise RuntimeError(f"Error al guardar producto: {e}")

        async def save_slider_file(file: UploadFile) -> str:
            try:
                temp_path = os.path.join(UPLOAD_FOLDER, generate_uuid_filename("tmp"))
                with open(temp_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                image = Image.open(temp_path)
                if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
                    image = image.convert("RGBA")
                else:
                    image = image.convert("RGB")
                image = ImageOps.fit(image, (1500, 500), method=Image.LANCZOS, centering=(0.5, 0.5))
                filename = generate_uuid_filename("webp")
                final_path = os.path.join(UPLOAD_FOLDER, filename)
                image.save(final_path, "WEBP", quality=100, lossless=True)
                os.remove(temp_path)
                return f"/{final_path}"
            except Exception as e:
                raise RuntimeError(f"Error al guardar slider: {e}")

        def delete_file(path: str) -> None:
            try:
                full_path = path.lstrip("/")
                if os.path.exists(full_path):
                    os.remove(full_path)
            except Exception as e:
                raise RuntimeError(f"Error al eliminar el archivo: {e}")
    '''), encoding="utf-8")
    (core_path / "security.py").write_text(dedent('''\
        from datetime import datetime, timedelta
        from typing import Optional
        from jose import jwt, JWTError
        from core.config import JWT_SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
        from passlib.context import CryptContext

        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

        def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
            to_encode = data.copy()
            expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
            to_encode.update({"exp": expire})
            return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=ALGORITHM)

        def decode_token(token: str) -> Optional[dict]:
            try:
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
                return payload
            except JWTError:
                return None
    '''), encoding="utf-8")
    (core_path / "logger.py").write_text(dedent('''\
            import logging
            from logging.handlers import RotatingFileHandler
            import os
            from pathlib import Path

            # Asegurar carpeta logs
            LOG_DIR = Path("logs")
            LOG_DIR.mkdir(parents=True, exist_ok=True)

            LOG_FILE = LOG_DIR / "app.log"

            # Crear logger
            logger = logging.getLogger("auth_logger")
            logger.setLevel(logging.INFO)

            # Evitar duplicados si se reimporta
            if not logger.handlers:
                # Rotating file handler
                file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=5)
                file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)

                # (Opcional) Consola
                console = logging.StreamHandler()
                console.setFormatter(file_formatter)
                logger.addHandler(console)
        '''), encoding="utf-8")
    print(f"✓ Módulos core generados en {core_path.resolve()}")
def crear_repositorio_auth(base_dir: str = "."):
    """
    Crea el archivo repositorio/auth.py con la implementación de AuthRepository.
    """
    repo_dir = Path(base_dir) / "repositorios"
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "__init__.py").write_text("", encoding="utf-8")

    contenido = dedent('''\
        from datetime import datetime
        from typing import Optional

        from sqlalchemy import select, update
        from sqlalchemy.ext.asyncio import AsyncSession

        from modelos.usuarios import Usuarios
        from core.security import pwd_context


        class AuthRepository:
            def __init__(self, session: AsyncSession):
                self.session = session
            async def login(self, username: str, password: str) -> Optional[Usuarios]:
                result = await self.session.execute(
                    select(Usuarios).where(Usuarios.username == username, Usuarios.borrado == 'no')
                )
                user = result.scalars().first()
                if user and pwd_context.verify(password, user.password):
                    return user
                return None
            async def get_by_email(self, email: str) -> Optional[Usuarios]:
                result = await self.session.execute(
                    select(Usuarios).where(Usuarios.email == email, Usuarios.borrado == 'no')
                )
                return result.scalars().first()

            async def get_by_username(self, username: str) -> Optional[Usuarios]:
                result = await self.session.execute(
                    select(Usuarios).where(Usuarios.username == username, Usuarios.borrado == 'no')
                )
                return result.scalars().first()

            async def update_password(self, user: Usuarios, hashed_password: str) -> bool:
                try:
                    await self.session.execute(
                        update(Usuarios)
                        .where(Usuarios.id == user.id)
                        .values(password=hashed_password)
                    )
                    await self.session.commit()
                    return True
                except Exception:
                    await self.session.rollback()
                    return False

            async def set_codigo_temporal(self, user: Usuarios, hashed_code: str) -> bool:
                try:
                    await self.session.execute(
                        update(Usuarios)
                        .where(Usuarios.id == user.id)
                        .values(codigo_temporal=hashed_code)
                    )
                    await self.session.commit()
                    return True
                except Exception:
                    await self.session.rollback()
                    return False

            def is_account_expired(self, user: Usuarios) -> bool:
                \"""
                Verifica si la cuenta está vencida según el campo `expira_el`.
                Si no tiene fecha de expiración, se considera activa.
                \"""
                if hasattr(user, "expira_el") and user.expira_el:
                    return user.expira_el < datetime.utcnow()
                return False
    ''')

    (repo_dir / "auth.py").write_text(contenido, encoding="utf-8")
    print(f"✓ Archivo AuthRepository creado en {repo_dir / 'auth.py'}")
def create_auth_service(base_dir: str = "."):
        """
        Crea el archivo servicios/auth_service.py con la implementación de AuthService.
        """
        servicios_dir = Path(base_dir) / "services"
        servicios_dir.mkdir(parents=True, exist_ok=True)

        service_code = dedent('''\
            import logging
            from typing import Optional
            import secrets
            import string
            from passlib.context import CryptContext
            from repositorios.auth import AuthRepository
            from modelos.usuarios import Usuario
            from core.security import create_access_token
            from tasks.email_tasks import enviar_email_recuperacion
            from fastapi import HTTPException

            logger = logging.getLogger(__name__)
            pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

            class AuthService:
                def __init__(self, repo: AuthRepository):
                    self.repo = repo

                async def login(self, username: str, password: str) -> Optional[str]:
                    """
                    Autentica al usuario. Si es válido y la cuenta está activa,
                    retorna un token JWT. Si no, retorna None.
                    """
                    try:
                        user: Usuario = await self.repo.get_by_username(username)
                        if not user:
                            logger.info(f"[AuthService] Login fallido para {username}")
                            return None

                        if not pwd_context.verify(password, user.password):
                            logger.warning(f"[AuthService] Contraseña incorrecta para {username}")
                            return None

                        if self.repo.is_account_expired(user):
                            logger.warning(f"[AuthService] Cuenta expirada para {username}")
                            return None

                        token_data = {
                            "sub": str(user.id),
                            "username": user.username,
                            "rol": user.rol
                        }
                        return create_access_token(token_data)
                    except Exception as e:
                        logger.error(f"[AuthService.login] Error inesperado: {e}")
                        return None

                async def enviar_codigo_recuperacion(self, username: str) -> bool:
                    """
                    Envía un código de recuperación de contraseña al email del usuario.
                    """
                    try:
                        usuario = await self.repo.get_by_username(username)

                        if not usuario:
                            logger.warning(f"[Recuperación] Usuario no encontrado: {username}")
                            return False

                        if not usuario.email:
                            logger.warning(f"[Recuperación] Usuario sin email registrado: {username}")
                            return False

                        # Generar clave temporal aleatoria
                        codigo = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(8))
                        hash_codigo = pwd_context.hash(codigo)

                        # Guardar el hash del código temporal
                        await self.repo.update_password(usuario, hash_codigo)

                        # Enviar email por cola asíncrona
                        enviar_email_recuperacion.send(usuario.email, codigo)
                        logger.info(f"[Recuperación] Código enviado a {usuario.email}")
                        return True
                    except Exception as e:
                        logger.error(f"[Recuperación] Error al enviar código a {username}: {e}")
                        raise HTTPException(status_code=500, detail="No se pudo enviar el código de recuperación")
        ''')

        service_file = servicios_dir / "auth_service.py"
        service_file.write_text(service_code, encoding="utf-8")
        print(f"✓ Servicio AuthService creado en {service_file.resolve()}")
def create_auth_controller(filename: str = "controladores/auth.py") -> None:
    """
    Genera el archivo de controlador de autenticación respetando
    el formato e imports provistos por el usuario.
    """
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    code = dedent('''\
    from fastapi import APIRouter, Depends, HTTPException, status
    from schemas.auth import LoginRequest, LoginResponse, OlvideClave
    from services.auth import AuthService
    from dependencies.auth import get_current_user
    from dependencies.dependencies import get_auth_service
    from dependencies.dependencies import get_usuarios_service
    from modelos.usuarios import Usuarios
    from services.usuarios_service import UsuariosService
    from typing import Optional
    from core.logger import logger
    from pydantic import BaseModel
    
    router = APIRouter()
    
    class PerfilResponse(BaseModel):
        id: int
        nombre: str
        apellido: str
        dni: str
        username: str
        email: str
        rol: str
        telefono: Optional[str]
        direccion: Optional[str]
        empresa: Optional[str]
        pais: str
        provincia: Optional[str]
        ciudad: Optional[str]
        codigo_postal: Optional[str]
        logo: Optional[str]
        numero_sucursales: int
        modulo_armado_caja: str
        modulo_sitio_web: str
        modulo_sistema_interno: str
        model_config = {
            "from_attributes": True
        }

    @router.post("/login", response_model=LoginResponse)
    async def login(
        data: LoginRequest,
        service: AuthService = Depends(get_auth_service)
    ):
        """
        Endpoint de autenticación de usuario.

        Recibe las credenciales (usuario y contraseña), verifica su validez
        a través del servicio de autenticación y devuelve un token JWT si es exitoso.

        - ✅ Devuelve un `access_token` si el login es correcto.
        - ❌ Devuelve un error 401 si el usuario o contraseña no coinciden.
        - ❌ Devuelve un error 500 si ocurre un error inesperado.

        Logs:
        - Login exitoso
        - Login fallido
        - Errores internos

        Args:
            data (LoginRequest): Datos de login enviados por el usuario.
            service (AuthService): Servicio de autenticación inyectado por FastAPI.

        Returns:
            dict: Token de acceso y tipo.
        """
        try:
            token = await service.login(data.username, data.password)
            if not token:
                logger.warning(f"LOGIN FALLIDO: Usuario: {data.username}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Usuario o contraseña inválidos"
                )
            logger.info(f"LOGIN EXITOSO: Usuario: {data.username}")
            return {"access_token": token, "token_type": "bearer"}
        except Exception as e:
            logger.exception(f"ERROR INTERNO EN LOGIN: Usuario: {data.username} - Error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Usuario o contraseña inválidos"
            )

    #34730603

    @router.post("/recuperar_clave", response_model=dict)
    async def recuperar_contrasena(
        request: OlvideClave,
        service: AuthService = Depends(get_auth_service)
    ):
        """
        Endpoint para solicitar recuperación de contraseña.

        Recibe un identificador de usuario (email o DNI), y si existe en el sistema,
        envía un código de recuperación al email asociado.

        - ✅ Devuelve un mensaje indicando que se envió un correo (si el usuario existe).
        - ❌ Devuelve un error 404 si el usuario no se encuentra.
        - ❌ Devuelve un error 500 si ocurre un error inesperado.

        Logs:
        - Recuperación exitosa
        - Usuario no encontrado
        - Errores internos

        Args:
            request (OlvideClave): Objeto con el campo `username` (email o identificador).
            service (AuthService): Servicio de autenticación inyectado por FastAPI.

        Returns:
            dict: Mensaje informando que el correo fue enviado.
        """
        try:
            success = await service.enviar_codigo_recuperacion(request.username)
            if not success:
                logger.warning(f"RECUPERACIÓN FALLIDA: Usuario no encontrado: {request.username}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No se encontró un usuario con ese email"
                )
            logger.info(f"RECUPERACIÓN EXITOSA: Email enviado a {request.username}")
            return {"msg": "Se envió un correo con instrucciones para recuperar la contraseña."}
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"ERROR INTERNO EN RECUPERACIÓN: Usuario: {request.username} - Error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Ocurrió un error inesperado al procesar la solicitud."
            )
            
    @router.get("/perfil", response_model=PerfilResponse)
    async def obtener_perfil(usuario: Usuarios = Depends(get_current_user)):
        """
        Devuelve el perfil del usuario autenticado.
        """
        try:
            logger.info(f"PERFIL CONSULTADO: {usuario.username} ({usuario.rol})")
            return usuario  # gracias a `orm_mode`, se convierte automáticamente
        except Exception as e:
            logger.exception(f"ERROR EN PERFIL: Usuario: {usuario.username} - {str(e)}")
            raise HTTPException(status_code=500, detail="Error al obtener perfil.")
    
    @router.post("/perfil")
    async def actualizar_perfil(
        update_data: dict,  # o un esquema si querés validación
        usuario: Usuarios = Depends(get_current_user),
        usuarios_service: UsuariosService = Depends(get_usuarios_service),
    
    ):
        try:
            if 'password' in update_data:
                if update_data['password']:
                    from passlib.context import CryptContext
                    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
                    update_data['password'] = pwd_context.hash(update_data['password'])
                else:
                    # Si es string vacío o None, eliminar el campo
                    del update_data['password']
    
            await usuarios_service.update_usuario(usuario.id, update_data)
            return {"mensaje": "Perfil actualizado correctamente."}
        except Exception as e:
            logger.exception(f"ERROR ACTUALIZANDO PERFIL {usuario.id}: {e}")
            raise HTTPException(status_code=500, detail="Error al actualizar perfil.")
    ''')

    Path(filename).write_text(code, encoding="utf-8")
    print(f"✓ Archivo {filename} creado con éxito.")
def crear_schema_auth(base_dir: str = "."):
    """
    Crea el archivo schemas/auth.py con los modelos Pydantic
    para autenticación.
    """
    schemas_dir = Path(base_dir) / "schemas"
    schemas_dir.mkdir(parents=True, exist_ok=True)
    (schemas_dir / "__init__.py").write_text("", encoding="utf-8")

    contenido = dedent('''\
    from pydantic import BaseModel

    class LoginRequest(BaseModel):
        username: str
        password: str

    class LoginResponse(BaseModel):
        access_token: str
        token_type: str = "bearer"

    class OlvideClave(BaseModel):
        username: str
    ''')

    (schemas_dir / "auth.py").write_text(contenido, encoding="utf-8")
    print(f"✓ Archivo schemas/auth.py creado en {schemas_dir / 'auth.py'}")
def create_email_tasks_module(filename: str = "tasks/email_tasks.py") -> None:
        """
        Crea el archivo tasks/email_tasks.py con actores Dramatiq para enviar correos.
        También asegura que tasks sea un paquete Python (crea __init__.py).
        """
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / "__init__.py").write_text("", encoding="utf-8")

        code = dedent('''\
            import dramatiq
            import smtplib
            import logging
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText
            from core.config import SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASSWORD

            logger = logging.getLogger(__name__)

            @dramatiq.actor
            def enviar_email_bienvenida(data: dict):
                mensaje = MIMEMultipart()
                mensaje['From'] = SMTP_USER
                mensaje['To'] = data["email"]
                mensaje['Subject'] = 'Bienvenido al sistema'

                html = f"""
                <html><body>
                <p>Estimado <strong>{data.get('nombre', 'Usuario')}</strong>,</p>
                <p>Bienvenido al sistema.</p>
                <p><strong>Empresa:</strong> {data.get('empresa')}</p>
                <p><strong>Tel:</strong> {data.get('tel')}</p>
                </body></html>
                """
                mensaje.attach(MIMEText(html, 'html'))

                try:
                    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as servidor:
                        servidor.starttls()
                        servidor.login(SMTP_USER, SMTP_PASSWORD)
                        servidor.sendmail(SMTP_USER, data["email"], mensaje.as_string())
                    logger.info(f"[Email enviado] a {data['email']}")
                except Exception as e:
                    logger.error(f"[ERROR al enviar email a {data['email']}] {e}")


            @dramatiq.actor
            def enviar_email_recuperacion(destinatario: str, codigo: str):
                try:
                    asunto = "Recuperación de contraseña"
                    mensaje_html = f"""
                    <html>
                    <body>
                        <p>Hola,</p>
                        <p>Recibimos una solicitud para recuperar tu contraseña.</p>
                        <p>Tu código de recuperación es: <strong>{codigo}</strong></p>
                        <p>Si no solicitaste esto, puedes ignorar este mensaje.</p>
                    </body>
                    </html>
                    """

                    mensaje = MIMEMultipart()
                    mensaje['From'] = SMTP_USER
                    mensaje['To'] = destinatario
                    mensaje['Subject'] = asunto
                    mensaje.attach(MIMEText(mensaje_html, 'html'))

                    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                        server.starttls()
                        server.login(SMTP_USER, SMTP_PASSWORD)
                        server.sendmail(SMTP_USER, destinatario, mensaje.as_string())

                    logger.info(f"[enviar_email_recuperacion] Email enviado a {destinatario}")

                except Exception as e:
                    logger.error(f"[enviar_email_recuperacion] Error al enviar email: {e}")
        ''')

        path.write_text(code, encoding="utf-8")
        print(f"✓ Archivo {filename} creado con éxito.")
def crear_service_auth(base_dir: str = "."):
    """
    Crea el archivo servicios/auth.py con la implementación de AuthService.
    También garantiza que 'servicios' sea un paquete (__init__.py).
    """
    servicios_dir = Path(base_dir) / "services"
    servicios_dir.mkdir(parents=True, exist_ok=True)
    contenido = dedent('''\
        import logging
        from fastapi import HTTPException
        from typing import Optional
        from passlib.context import CryptContext
        import secrets
        import string
        from repositorios.auth import AuthRepository
        from modelos.usuarios import Usuarios
        from core.security import create_access_token
        from tasks.email_tasks import enviar_email_recuperacion

        logger = logging.getLogger(__name__)
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

        class AuthService:
            def __init__(self, repo: AuthRepository):
                self.repo = repo

            async def login(self, username: str, password: str) -> Optional[str]:
                """
                Autentica al usuario. Si es válido y la cuenta está activa,
                retorna un token JWT. Si no, retorna None.
                """
                try:
                    user: Usuarios = await self.repo.get_by_username(username)
                    if not user:
                        logger.info(f"[AuthService] Login fallido para {username}")
                        return None

                    # Verificar hash de contraseña
                    if not pwd_context.verify(password, user.password):
                        logger.warning(f"[AuthService] Contraseña incorrecta para {username}")
                        return None

                    if self.repo.is_account_expired(user):
                        logger.warning(f"[AuthService] Cuenta expirada para {username}")
                        return None

                    token_data = {
                        "sub": str(user.id),
                        "username": user.username,
                        "rol": user.rol
                    }
                    return create_access_token(token_data)
                except Exception as e:
                    logger.error(f"[AuthService.login] Error inesperado: {e}")
                    return None

            async def enviar_codigo_recuperacion(self, username: str) -> bool:
                try:
                    usuario = await self.repo.get_by_username(username)

                    if not usuario:
                        logger.warning(f"[Recuperación] Usuario no encontrado: {username}")
                        return False

                    if not usuario.email:
                        logger.warning(f"[Recuperación] Usuario sin email registrado: {username}")
                        return False

                    # Generar clave temporal aleatoria
                    codigo = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(8))
                    hash_codigo = pwd_context.hash(codigo)

                    # Guardar el hash del código temporal
                    await self.repo.update_password(usuario, hash_codigo)

                    # Enviar email por cola asíncrona (ej. Dramatiq)
                    enviar_email_recuperacion.send(usuario.email, codigo)
                    logger.info(f"[Recuperación] Código enviado a {usuario.email}")
                    return True
                except Exception as e:
                    logger.error(f"[Recuperación] Error al enviar código a {username}: {e}")
                    raise HTTPException(status_code=500, detail="No se pudo enviar el código de recuperación")
    ''')
    (servicios_dir / "auth.py").write_text(contenido, encoding="utf-8")
    print(f"✓ Archivo AuthService creado en {servicios_dir / 'auth.py'}")
def _ensure_main_entry(
    project_name: str = "Techware",
    controllers_pkg_candidates: list[str] = ("controladores"),
    filename: str = "main.py",
    base_api_prefix: str = "/api",
    static_mount: str = "/uploads",
    static_dir: str = "uploads",
    cors_origins: list[str] | None = None,
):
    """
    Crea un entry FastAPI con meta-descubrimiento de routers.
    - Busca routers en los paquetes candidatos (por defecto: 'controladores', 'controllers').
    - Incluye cualquier atributo APIRouter exportado (router, *_router, admin_router, ...).
    - Monta /uploads y aplica CORS.
    """
    import importlib
    import pathlib as _pl
    from textwrap import dedent

    cors_origins = cors_origins or ["*"]

    # Descubrir paquete de controladores
    pkg_name = None
    for cand in controllers_pkg_candidates:
        try:
            importlib.import_module(cand)
            pkg_name = cand
            break
        except Exception:
            continue

    # Si no existe ninguno, crear 'controllers' vacío como fallback
    if not pkg_name:
        _pl.Path("controllers").mkdir(parents=True, exist_ok=True)
        (_pl.Path("controllers") / "__init__.py").write_text("", encoding="utf-8")
        pkg_name = "controllers"

    code = f'''\
from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import importlib, pkgutil, inspect

# === Config dinámica ===
PROJECT_NAME = {project_name!r}
BASE_API_PREFIX = {base_api_prefix!r}
STATIC_MOUNT = {static_mount!r}
STATIC_DIR = {static_dir!r}
CORS_ORIGINS = {cors_origins!r}
CONTROLLERS_PKG = {pkg_name!r}

app = FastAPI(
    title=f"{{PROJECT_NAME}} API",
    version="1.0.0",
    description=f"Backend de {{PROJECT_NAME}} con autenticación JWT y estructura limpia"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static /uploads
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = (BASE_DIR / STATIC_DIR)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
app.mount(STATIC_MOUNT, StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

def _iter_modules(package_name: str):
    pkg = importlib.import_module(package_name)
    if not hasattr(pkg, "__path__"):
        return
    for m in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        yield m.name

def _is_auth_module(mod_name: str) -> bool:
    tail = mod_name.rsplit(".", 1)[-1].lower()
    return tail in ("auth", "authentication", "login", "oauth")

def _include_all_routers():
    try:
        from fastapi import APIRouter  # type: ignore
    except Exception:
        return
    seen = set()
    for mod_name in _iter_modules(CONTROLLERS_PKG) or []:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        # buscar cualquier atributo que sea APIRouter (duck-typing + nombre de clase)
        for attr_name, val in vars(mod).items():
            try:
                cls_name = getattr(getattr(val, "__class__", None), "__name__", None)
                is_router = (cls_name == "APIRouter") or hasattr(val, "routes")
            except Exception:
                is_router = False
            if not is_router:
                continue
            router = val
            if id(router) in seen:
                continue
            seen.add(id(router))
            # prefijo: auth -> /api/auth, el resto -> /api
            base_prefix = BASE_API_PREFIX + "/auth" if _is_auth_module(mod_name) else BASE_API_PREFIX
            try:
                app.include_router(router, prefix=base_prefix)
            except Exception:
                # fallback por si el router ya trae su propio prefix
                app.include_router(router)

# incluir routers autodescubiertos
_include_all_routers()

# Raíz
@app.get("/")
async def root():
    return {{"mensaje": f"Bienvenido a la API de {{PROJECT_NAME}} 🚀"}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        port=8000,
        reload=True
    )
'''
    _pl.Path(filename).write_text(dedent(code), encoding="utf-8")
    print(f"✓ {filename} generado con meta-routers (paquete: {pkg_name})")



import importlib
import inspect
import pkgutil
import re
from pathlib import Path

def _discover_classes(pkg_name: str, suffix: str):
    """
    Devuelve dict {BaseName: (ClassName, module_path)} para clases que terminan en suffix.
    BaseName = nombre sin el suffix. p.ej.: UsuarioService -> 'Usuario'
    """
    out = {}
    try:
        pkg = importlib.import_module(pkg_name)
    except Exception:
        return out
    if not hasattr(pkg, "__path__"):
        return out
    for m in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        try:
            mod = importlib.import_module(m.name)
        except Exception:
            continue
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if name.endswith(suffix) and obj.__module__ == mod.__name__:
                base = name[: -len(suffix)]
                out[base] = (name, mod.__name__)
    return out

def _snake(name: str) -> str:
    """CamelCase -> snake_case"""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

def _flat_snake(name: str) -> str:
    """CamelCase -> snake_case sin guiones: MedioDeEnvio -> mediodeenvio"""
    return _snake(name).replace("_", "")

def generate_dependencies_strict(
    deps_pkg_dir: str = "dependencies",
    services_pkg: str = "services",
    repos_pkg: str = "repositorios",
):
    """
    Genera dependencies/dependencies.py con:
      - header idéntico a tu ejemplo (incluye AsyncGenerator y async_session)
      - imports explícitos por módulo (repos primero, luego services)
      - un provider get_<alias>_service por cada *Service encontrado
      - alias por defecto: nombre 'aplanado' sin guiones (MedioDePago -> get_mediodepago_service)
      - overrides para casos especiales (AbonoMensual -> get_abono_service)
    """
    services = _discover_classes(services_pkg, "Service")
    repos    = _discover_classes(repos_pkg, "Repository")

    # overrides de nombre del provider (clave = BaseName del Service)
    provider_overrides = {
        "AbonoMensual": "abono",
        "MedioDeEnvio": "mediodeenvio",
        "MedioDePago":  "mediodepago",
        # agregá acá los que necesites
    }

    # armo imports por módulo
    repo_imports_by_module = {}
    for _base, (cls, module) in repos.items():
        repo_imports_by_module.setdefault(module, []).append(cls)

    svc_imports_by_module = {}
    for _base, (cls, module) in services.items():
        svc_imports_by_module.setdefault(module, []).append(cls)

    lines = []
    # header como tu ejemplo
    lines.append("from typing import AsyncGenerator\n\n")
    lines.append("from fastapi import Depends\n")
    lines.append("from sqlalchemy.ext.asyncio import AsyncSession\n")
    lines.append("from database import get_async_session, async_session\n\n")
    lines.append("# Importá tus repos y services\n")

    # imports de repos (orden estable)
    for module in sorted(repo_imports_by_module.keys()):
        classes = ", ".join(sorted(repo_imports_by_module[module]))
        lines.append(f"from {module} import {classes}\n")

    # imports de services (orden estable)
    for module in sorted(svc_imports_by_module.keys()):
        classes = ", ".join(sorted(svc_imports_by_module[module]))
        lines.append(f"from {module} import {classes}\n")

    # Si tenés Auth en paquetes “auth”, estos imports ya entran arriba.
    lines.append("\n# ---------- Servicios ----------\n\n\n")

    # providers
    for base in sorted(services.keys()):
        svc_cls, _svc_mod = services[base]
        repo_info = repos.get(base)
        # nombre del provider: override -> aplanado -> snake si querés
        alias = provider_overrides.get(base) or _flat_snake(base)
        provider_name = f"get_{alias}_service"

        lines.append(f"async def {provider_name}(session: AsyncSession = Depends(get_async_session)) -> {svc_cls}:\n")
        if repo_info:
            repo_cls, _repo_mod = repo_info
            lines.append(f"    repo = {repo_cls}(session)\n")
            lines.append(f"    return {svc_cls}(repo)\n\n")
        else:
            # si no hay repo correspondiente, devolvemos el service con None
            lines.append(f"    # TODO: No se encontró {base}Repository para {svc_cls}\n")
            lines.append(f"    return {svc_cls}(repo=None)  # type: ignore\n\n")

    # escribir archivo
    deps_dir = Path(deps_pkg_dir)
    deps_dir.mkdir(parents=True, exist_ok=True)
    (deps_dir / "__init__.py").write_text("", encoding="utf-8")
    (deps_dir / "dependencies.py").write_text("".join(lines), encoding="utf-8")
    print(f"✓ dependencies/dependencies.py generado con {len(services)} providers")

def build_services_init(services_dir: str = "services"):
    """
    Genera services/__init__.py con:
      - import de cada clase *Service encontrada dentro del paquete services
      - __all__ con la lista de todas las clases *Service
    Busca módulos recursivamente dentro de `services/`.

    Reglas:
      - Sólo considera clases cuyo nombre termina en 'Service'
      - Evita falsos positivos chequeando que __module__ coincida con el módulo
      - Orden alfabético estable para imports y __all__
    """
    pkg_path = Path(services_dir)
    pkg_path.mkdir(parents=True, exist_ok=True)

    # Asegurar que sea paquete
    init_file = pkg_path / "__init__.py"
    if not init_file.exists():
        init_file.write_text("", encoding="utf-8")

    pkg_name = services_dir.replace("/", ".").rstrip(".")

    # Descubrir módulos del paquete
    try:
        pkg = importlib.import_module(pkg_name)
    except Exception as e:
        raise RuntimeError(f"No pude importar el paquete '{pkg_name}': {e}")

    modules = []
    if hasattr(pkg, "__path__"):
        for m in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
            # ignorar el propio __init__
            if m.name.endswith(".__init__"):
                continue
            modules.append(m.name)

    # Mapear: modulo_relativo -> [ClasesService...]
    services_by_module: dict[str, list[str]] = {}

    for mod_name in sorted(modules):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            # si un módulo falla al importar, lo saltamos
            continue

        classes = []
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if not name.endswith("Service"):
                continue
            # evitar clases importadas desde otro lado
            if getattr(obj, "__module__", None) != mod.__name__:
                continue
            classes.append(name)

        if classes:
            rel_module = mod_name.split(".", 1)[1]  # 'services.foo_service' -> 'foo_service'
            rel_module = rel_module.split(".", 1)[1] if rel_module.startswith("services.") else rel_module
            services_by_module[rel_module] = sorted(classes)

    # Construir contenido de __init__.py
    lines: list[str] = ["# Auto-generado. Exporta services.\n"]
    all_classes: list[str] = []

    for rel_module in sorted(services_by_module.keys()):
        class_list = services_by_module[rel_module]
        all_classes.extend(class_list)
        # Import relativo: from .archivo import Clase1, Clase2
        # rel_module puede incluir submódulos (p.ej. "subpkg.mod"); ante eso usamos from .subpkg.mod import ...
        lines.append(f"from .{rel_module} import {', '.join(class_list)}\n")

    lines.append("\n__all__ = [\n")
    for cls in sorted(all_classes):
        lines.append(f"    '{cls}',\n")
    lines.append("]\n")

    init_file.write_text("".join(lines), encoding="utf-8")
    print(f"✓ services/__init__.py actualizado ({len(all_classes)} services)")





@app.command(help="Hace TODO: genera modelos, repos, services y schemas")
def scaffold(models_dir: str = "modelos", repos_dir: str = "repositorios", services_dir: str = "services",
             schemas_dir: str = "schemas", project_name: str = "Techware"):
    generate_models(out_dir=models_dir)
    models_pkg = models_dir.replace("/", ".")
    generate_repositories(models_pkg=models_pkg, out_dir=repos_dir, repos_base_pkg=f"{repos_dir}.base")
    generate_services(models_pkg=models_pkg, repos_pkg=repos_dir.replace("/", "."), out_dir=services_dir)
    generate_schemas(models_pkg=models_pkg, out_dir=schemas_dir)
    _ensure_database_py()
    _ensure_dependencies_pkg_dynamic()
    crear_repositorio_auth()
    create_auth_controller()
    generate_controllers(
        services_pkg=services_dir.replace("/", "."),
        roles_yaml="roles_controlador.yaml",
        out_dir="controladores",
        models_pkg=models_pkg,
    )
    crear_schema_auth()
    create_email_tasks_module()
    ensure_core_modules()
    crear_service_auth()
    _ensure_main_entry(
        project_name=project_name,
        controllers_pkg_candidates=["controladores"],
        filename="main.py",  # Cambiá a generador_base.py si querés
        base_api_prefix="/api",
        static_mount="/uploads",
        static_dir="uploads",
        cors_origins=["*"],  # en prod: poné orígenes concretos
    )
    generate_dependencies_strict(
        deps_pkg_dir="dependencies",
        services_pkg="services",
        repos_pkg="repositorios",
    )
    build_services_init()
if __name__ == "__main__":
    app()
