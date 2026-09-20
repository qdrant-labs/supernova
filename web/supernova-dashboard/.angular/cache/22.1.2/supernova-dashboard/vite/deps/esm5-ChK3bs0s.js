//#region node_modules/tslib/tslib.es6.mjs
/******************************************************************************
Copyright (c) Microsoft Corporation.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
***************************************************************************** */
var extendStatics = function(d, b) {
	extendStatics = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(d, b) {
		d.__proto__ = b;
	} || function(d, b) {
		for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p];
	};
	return extendStatics(d, b);
};
function __extends(d, b) {
	if (typeof b !== "function" && b !== null) throw new TypeError("Class extends value " + String(b) + " is not a constructor or null");
	extendStatics(d, b);
	function __() {
		this.constructor = d;
	}
	d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
}
function __awaiter(thisArg, _arguments, P, generator) {
	function adopt(value) {
		return value instanceof P ? value : new P(function(resolve) {
			resolve(value);
		});
	}
	return new (P || (P = Promise))(function(resolve, reject) {
		function fulfilled(value) {
			try {
				step(generator.next(value));
			} catch (e) {
				reject(e);
			}
		}
		function rejected(value) {
			try {
				step(generator["throw"](value));
			} catch (e) {
				reject(e);
			}
		}
		function step(result) {
			result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected);
		}
		step((generator = generator.apply(thisArg, _arguments || [])).next());
	});
}
function __generator(thisArg, body) {
	var _ = {
		label: 0,
		sent: function() {
			if (t[0] & 1) throw t[1];
			return t[1];
		},
		trys: [],
		ops: []
	}, f, y, t, g = Object.create((typeof Iterator === "function" ? Iterator : Object).prototype);
	return g.next = verb(0), g["throw"] = verb(1), g["return"] = verb(2), typeof Symbol === "function" && (g[Symbol.iterator] = function() {
		return this;
	}), g;
	function verb(n) {
		return function(v) {
			return step([n, v]);
		};
	}
	function step(op) {
		if (f) throw new TypeError("Generator is already executing.");
		while (g && (g = 0, op[0] && (_ = 0)), _) try {
			if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
			if (y = 0, t) op = [op[0] & 2, t.value];
			switch (op[0]) {
				case 0:
				case 1:
					t = op;
					break;
				case 4:
					_.label++;
					return {
						value: op[1],
						done: false
					};
				case 5:
					_.label++;
					y = op[1];
					op = [0];
					continue;
				case 7:
					op = _.ops.pop();
					_.trys.pop();
					continue;
				default:
					if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) {
						_ = 0;
						continue;
					}
					if (op[0] === 3 && (!t || op[1] > t[0] && op[1] < t[3])) {
						_.label = op[1];
						break;
					}
					if (op[0] === 6 && _.label < t[1]) {
						_.label = t[1];
						t = op;
						break;
					}
					if (t && _.label < t[2]) {
						_.label = t[2];
						_.ops.push(op);
						break;
					}
					if (t[2]) _.ops.pop();
					_.trys.pop();
					continue;
			}
			op = body.call(thisArg, _);
		} catch (e) {
			op = [6, e];
			y = 0;
		} finally {
			f = t = 0;
		}
		if (op[0] & 5) throw op[1];
		return {
			value: op[0] ? op[1] : void 0,
			done: true
		};
	}
}
function __values(o) {
	var s = typeof Symbol === "function" && Symbol.iterator, m = s && o[s], i = 0;
	if (m) return m.call(o);
	if (o && typeof o.length === "number") return { next: function() {
		if (o && i >= o.length) o = void 0;
		return {
			value: o && o[i++],
			done: !o
		};
	} };
	throw new TypeError(s ? "Object is not iterable." : "Symbol.iterator is not defined.");
}
function __read(o, n) {
	var m = typeof Symbol === "function" && o[Symbol.iterator];
	if (!m) return o;
	var i = m.call(o), r, ar = [], e;
	try {
		while ((n === void 0 || n-- > 0) && !(r = i.next()).done) ar.push(r.value);
	} catch (error) {
		e = { error };
	} finally {
		try {
			if (r && !r.done && (m = i["return"])) m.call(i);
		} finally {
			if (e) throw e.error;
		}
	}
	return ar;
}
function __spreadArray(to, from, pack) {
	if (pack || arguments.length === 2) {
		for (var i = 0, l = from.length, ar; i < l; i++) if (ar || !(i in from)) {
			if (!ar) ar = Array.prototype.slice.call(from, 0, i);
			ar[i] = from[i];
		}
	}
	return to.concat(ar || Array.prototype.slice.call(from));
}
function __await(v) {
	return this instanceof __await ? (this.v = v, this) : new __await(v);
}
function __asyncGenerator(thisArg, _arguments, generator) {
	if (!Symbol.asyncIterator) throw new TypeError("Symbol.asyncIterator is not defined.");
	var g = generator.apply(thisArg, _arguments || []), i, q = [];
	return i = Object.create((typeof AsyncIterator === "function" ? AsyncIterator : Object).prototype), verb("next"), verb("throw"), verb("return", awaitReturn), i[Symbol.asyncIterator] = function() {
		return this;
	}, i;
	function awaitReturn(f) {
		return function(v) {
			return Promise.resolve(v).then(f, reject);
		};
	}
	function verb(n, f) {
		if (g[n]) {
			i[n] = function(v) {
				return new Promise(function(a, b) {
					q.push([
						n,
						v,
						a,
						b
					]) > 1 || resume(n, v);
				});
			};
			if (f) i[n] = f(i[n]);
		}
	}
	function resume(n, v) {
		try {
			step(g[n](v));
		} catch (e) {
			settle(q[0][3], e);
		}
	}
	function step(r) {
		r.value instanceof __await ? Promise.resolve(r.value.v).then(fulfill, reject) : settle(q[0][2], r);
	}
	function fulfill(value) {
		resume("next", value);
	}
	function reject(value) {
		resume("throw", value);
	}
	function settle(f, v) {
		if (f(v), q.shift(), q.length) resume(q[0][0], q[0][1]);
	}
}
function __asyncValues(o) {
	if (!Symbol.asyncIterator) throw new TypeError("Symbol.asyncIterator is not defined.");
	var m = o[Symbol.asyncIterator], i;
	return m ? m.call(o) : (o = typeof __values === "function" ? __values(o) : o[Symbol.iterator](), i = {}, verb("next"), verb("throw"), verb("return"), i[Symbol.asyncIterator] = function() {
		return this;
	}, i);
	function verb(n) {
		i[n] = o[n] && function(v) {
			return new Promise(function(resolve, reject) {
				v = o[n](v), settle(resolve, reject, v.done, v.value);
			});
		};
	}
	function settle(resolve, reject, d, v) {
		Promise.resolve(v).then(function(v) {
			resolve({
				value: v,
				done: d
			});
		}, reject);
	}
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/isFunction.js
function isFunction(value) {
	return typeof value === "function";
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/createErrorClass.js
function createErrorClass(createImpl) {
	var _super = function(instance) {
		Error.call(instance);
		instance.stack = (/* @__PURE__ */ new Error()).stack;
	};
	var ctorFunc = createImpl(_super);
	ctorFunc.prototype = Object.create(Error.prototype);
	ctorFunc.prototype.constructor = ctorFunc;
	return ctorFunc;
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/UnsubscriptionError.js
var UnsubscriptionError = createErrorClass(function(_super) {
	return function UnsubscriptionErrorImpl(errors) {
		_super(this);
		this.message = errors ? errors.length + " errors occurred during unsubscription:\n" + errors.map(function(err, i) {
			return i + 1 + ") " + err.toString();
		}).join("\n  ") : "";
		this.name = "UnsubscriptionError";
		this.errors = errors;
	};
});
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/arrRemove.js
function arrRemove(arr, item) {
	if (arr) {
		var index = arr.indexOf(item);
		0 <= index && arr.splice(index, 1);
	}
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/Subscription.js
var Subscription = function() {
	function Subscription(initialTeardown) {
		this.initialTeardown = initialTeardown;
		this.closed = false;
		this._parentage = null;
		this._finalizers = null;
	}
	Subscription.prototype.unsubscribe = function() {
		var e_1, _a, e_2, _b;
		var errors;
		if (!this.closed) {
			this.closed = true;
			var _parentage = this._parentage;
			if (_parentage) {
				this._parentage = null;
				if (Array.isArray(_parentage)) try {
					for (var _parentage_1 = __values(_parentage), _parentage_1_1 = _parentage_1.next(); !_parentage_1_1.done; _parentage_1_1 = _parentage_1.next()) _parentage_1_1.value.remove(this);
				} catch (e_1_1) {
					e_1 = { error: e_1_1 };
				} finally {
					try {
						if (_parentage_1_1 && !_parentage_1_1.done && (_a = _parentage_1.return)) _a.call(_parentage_1);
					} finally {
						if (e_1) throw e_1.error;
					}
				}
				else _parentage.remove(this);
			}
			var initialFinalizer = this.initialTeardown;
			if (isFunction(initialFinalizer)) try {
				initialFinalizer();
			} catch (e) {
				errors = e instanceof UnsubscriptionError ? e.errors : [e];
			}
			var _finalizers = this._finalizers;
			if (_finalizers) {
				this._finalizers = null;
				try {
					for (var _finalizers_1 = __values(_finalizers), _finalizers_1_1 = _finalizers_1.next(); !_finalizers_1_1.done; _finalizers_1_1 = _finalizers_1.next()) {
						var finalizer = _finalizers_1_1.value;
						try {
							execFinalizer(finalizer);
						} catch (err) {
							errors = errors !== null && errors !== void 0 ? errors : [];
							if (err instanceof UnsubscriptionError) errors = __spreadArray(__spreadArray([], __read(errors)), __read(err.errors));
							else errors.push(err);
						}
					}
				} catch (e_2_1) {
					e_2 = { error: e_2_1 };
				} finally {
					try {
						if (_finalizers_1_1 && !_finalizers_1_1.done && (_b = _finalizers_1.return)) _b.call(_finalizers_1);
					} finally {
						if (e_2) throw e_2.error;
					}
				}
			}
			if (errors) throw new UnsubscriptionError(errors);
		}
	};
	Subscription.prototype.add = function(teardown) {
		var _a;
		if (teardown && teardown !== this) if (this.closed) execFinalizer(teardown);
		else {
			if (teardown instanceof Subscription) {
				if (teardown.closed || teardown._hasParent(this)) return;
				teardown._addParent(this);
			}
			(this._finalizers = (_a = this._finalizers) !== null && _a !== void 0 ? _a : []).push(teardown);
		}
	};
	Subscription.prototype._hasParent = function(parent) {
		var _parentage = this._parentage;
		return _parentage === parent || Array.isArray(_parentage) && _parentage.includes(parent);
	};
	Subscription.prototype._addParent = function(parent) {
		var _parentage = this._parentage;
		this._parentage = Array.isArray(_parentage) ? (_parentage.push(parent), _parentage) : _parentage ? [_parentage, parent] : parent;
	};
	Subscription.prototype._removeParent = function(parent) {
		var _parentage = this._parentage;
		if (_parentage === parent) this._parentage = null;
		else if (Array.isArray(_parentage)) arrRemove(_parentage, parent);
	};
	Subscription.prototype.remove = function(teardown) {
		var _finalizers = this._finalizers;
		_finalizers && arrRemove(_finalizers, teardown);
		if (teardown instanceof Subscription) teardown._removeParent(this);
	};
	Subscription.EMPTY = (function() {
		var empty = new Subscription();
		empty.closed = true;
		return empty;
	})();
	return Subscription;
}();
var EMPTY_SUBSCRIPTION = Subscription.EMPTY;
function isSubscription(value) {
	return value instanceof Subscription || value && "closed" in value && isFunction(value.remove) && isFunction(value.add) && isFunction(value.unsubscribe);
}
function execFinalizer(finalizer) {
	if (isFunction(finalizer)) finalizer();
	else finalizer.unsubscribe();
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/config.js
var config = {
	onUnhandledError: null,
	onStoppedNotification: null,
	Promise: void 0,
	useDeprecatedSynchronousErrorHandling: false,
	useDeprecatedNextContext: false
};
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/scheduler/timeoutProvider.js
var timeoutProvider = {
	setTimeout: function(handler, timeout) {
		var args = [];
		for (var _i = 2; _i < arguments.length; _i++) args[_i - 2] = arguments[_i];
		var delegate = timeoutProvider.delegate;
		if (delegate === null || delegate === void 0 ? void 0 : delegate.setTimeout) return delegate.setTimeout.apply(delegate, __spreadArray([handler, timeout], __read(args)));
		return setTimeout.apply(void 0, __spreadArray([handler, timeout], __read(args)));
	},
	clearTimeout: function(handle) {
		var delegate = timeoutProvider.delegate;
		return ((delegate === null || delegate === void 0 ? void 0 : delegate.clearTimeout) || clearTimeout)(handle);
	},
	delegate: void 0
};
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/reportUnhandledError.js
function reportUnhandledError(err) {
	timeoutProvider.setTimeout(function() {
		var onUnhandledError = config.onUnhandledError;
		if (onUnhandledError) onUnhandledError(err);
		else throw err;
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/noop.js
function noop() {}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/NotificationFactories.js
var COMPLETE_NOTIFICATION = (function() {
	return createNotification("C", void 0, void 0);
})();
function errorNotification(error) {
	return createNotification("E", void 0, error);
}
function nextNotification(value) {
	return createNotification("N", value, void 0);
}
function createNotification(kind, value, error) {
	return {
		kind,
		value,
		error
	};
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/errorContext.js
var context = null;
function errorContext(cb) {
	if (config.useDeprecatedSynchronousErrorHandling) {
		var isRoot = !context;
		if (isRoot) context = {
			errorThrown: false,
			error: null
		};
		cb();
		if (isRoot) {
			var _a = context, errorThrown = _a.errorThrown, error = _a.error;
			context = null;
			if (errorThrown) throw error;
		}
	} else cb();
}
function captureError(err) {
	if (config.useDeprecatedSynchronousErrorHandling && context) {
		context.errorThrown = true;
		context.error = err;
	}
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/Subscriber.js
var Subscriber = function(_super) {
	__extends(Subscriber, _super);
	function Subscriber(destination) {
		var _this = _super.call(this) || this;
		_this.isStopped = false;
		if (destination) {
			_this.destination = destination;
			if (isSubscription(destination)) destination.add(_this);
		} else _this.destination = EMPTY_OBSERVER;
		return _this;
	}
	Subscriber.create = function(next, error, complete) {
		return new SafeSubscriber(next, error, complete);
	};
	Subscriber.prototype.next = function(value) {
		if (this.isStopped) handleStoppedNotification(nextNotification(value), this);
		else this._next(value);
	};
	Subscriber.prototype.error = function(err) {
		if (this.isStopped) handleStoppedNotification(errorNotification(err), this);
		else {
			this.isStopped = true;
			this._error(err);
		}
	};
	Subscriber.prototype.complete = function() {
		if (this.isStopped) handleStoppedNotification(COMPLETE_NOTIFICATION, this);
		else {
			this.isStopped = true;
			this._complete();
		}
	};
	Subscriber.prototype.unsubscribe = function() {
		if (!this.closed) {
			this.isStopped = true;
			_super.prototype.unsubscribe.call(this);
			this.destination = null;
		}
	};
	Subscriber.prototype._next = function(value) {
		this.destination.next(value);
	};
	Subscriber.prototype._error = function(err) {
		try {
			this.destination.error(err);
		} finally {
			this.unsubscribe();
		}
	};
	Subscriber.prototype._complete = function() {
		try {
			this.destination.complete();
		} finally {
			this.unsubscribe();
		}
	};
	return Subscriber;
}(Subscription);
var _bind = Function.prototype.bind;
function bind(fn, thisArg) {
	return _bind.call(fn, thisArg);
}
var ConsumerObserver = function() {
	function ConsumerObserver(partialObserver) {
		this.partialObserver = partialObserver;
	}
	ConsumerObserver.prototype.next = function(value) {
		var partialObserver = this.partialObserver;
		if (partialObserver.next) try {
			partialObserver.next(value);
		} catch (error) {
			handleUnhandledError(error);
		}
	};
	ConsumerObserver.prototype.error = function(err) {
		var partialObserver = this.partialObserver;
		if (partialObserver.error) try {
			partialObserver.error(err);
		} catch (error) {
			handleUnhandledError(error);
		}
		else handleUnhandledError(err);
	};
	ConsumerObserver.prototype.complete = function() {
		var partialObserver = this.partialObserver;
		if (partialObserver.complete) try {
			partialObserver.complete();
		} catch (error) {
			handleUnhandledError(error);
		}
	};
	return ConsumerObserver;
}();
var SafeSubscriber = function(_super) {
	__extends(SafeSubscriber, _super);
	function SafeSubscriber(observerOrNext, error, complete) {
		var _this = _super.call(this) || this;
		var partialObserver;
		if (isFunction(observerOrNext) || !observerOrNext) partialObserver = {
			next: observerOrNext !== null && observerOrNext !== void 0 ? observerOrNext : void 0,
			error: error !== null && error !== void 0 ? error : void 0,
			complete: complete !== null && complete !== void 0 ? complete : void 0
		};
		else {
			var context_1;
			if (_this && config.useDeprecatedNextContext) {
				context_1 = Object.create(observerOrNext);
				context_1.unsubscribe = function() {
					return _this.unsubscribe();
				};
				partialObserver = {
					next: observerOrNext.next && bind(observerOrNext.next, context_1),
					error: observerOrNext.error && bind(observerOrNext.error, context_1),
					complete: observerOrNext.complete && bind(observerOrNext.complete, context_1)
				};
			} else partialObserver = observerOrNext;
		}
		_this.destination = new ConsumerObserver(partialObserver);
		return _this;
	}
	return SafeSubscriber;
}(Subscriber);
function handleUnhandledError(error) {
	if (config.useDeprecatedSynchronousErrorHandling) captureError(error);
	else reportUnhandledError(error);
}
function defaultErrorHandler(err) {
	throw err;
}
function handleStoppedNotification(notification, subscriber) {
	var onStoppedNotification = config.onStoppedNotification;
	onStoppedNotification && timeoutProvider.setTimeout(function() {
		return onStoppedNotification(notification, subscriber);
	});
}
var EMPTY_OBSERVER = {
	closed: true,
	next: noop,
	error: defaultErrorHandler,
	complete: noop
};
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/symbol/observable.js
var observable = (function() {
	return typeof Symbol === "function" && Symbol.observable || "@@observable";
})();
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/identity.js
function identity(x) {
	return x;
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/pipe.js
function pipe() {
	var fns = [];
	for (var _i = 0; _i < arguments.length; _i++) fns[_i] = arguments[_i];
	return pipeFromArray(fns);
}
function pipeFromArray(fns) {
	if (fns.length === 0) return identity;
	if (fns.length === 1) return fns[0];
	return function piped(input) {
		return fns.reduce(function(prev, fn) {
			return fn(prev);
		}, input);
	};
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/Observable.js
var Observable = function() {
	function Observable(subscribe) {
		if (subscribe) this._subscribe = subscribe;
	}
	Observable.prototype.lift = function(operator) {
		var observable = new Observable();
		observable.source = this;
		observable.operator = operator;
		return observable;
	};
	Observable.prototype.subscribe = function(observerOrNext, error, complete) {
		var _this = this;
		var subscriber = isSubscriber(observerOrNext) ? observerOrNext : new SafeSubscriber(observerOrNext, error, complete);
		errorContext(function() {
			var _a = _this, operator = _a.operator, source = _a.source;
			subscriber.add(operator ? operator.call(subscriber, source) : source ? _this._subscribe(subscriber) : _this._trySubscribe(subscriber));
		});
		return subscriber;
	};
	Observable.prototype._trySubscribe = function(sink) {
		try {
			return this._subscribe(sink);
		} catch (err) {
			sink.error(err);
		}
	};
	Observable.prototype.forEach = function(next, promiseCtor) {
		var _this = this;
		promiseCtor = getPromiseCtor(promiseCtor);
		return new promiseCtor(function(resolve, reject) {
			var subscriber = new SafeSubscriber({
				next: function(value) {
					try {
						next(value);
					} catch (err) {
						reject(err);
						subscriber.unsubscribe();
					}
				},
				error: reject,
				complete: resolve
			});
			_this.subscribe(subscriber);
		});
	};
	Observable.prototype._subscribe = function(subscriber) {
		var _a;
		return (_a = this.source) === null || _a === void 0 ? void 0 : _a.subscribe(subscriber);
	};
	Observable.prototype[observable] = function() {
		return this;
	};
	Observable.prototype.pipe = function() {
		var operations = [];
		for (var _i = 0; _i < arguments.length; _i++) operations[_i] = arguments[_i];
		return pipeFromArray(operations)(this);
	};
	Observable.prototype.toPromise = function(promiseCtor) {
		var _this = this;
		promiseCtor = getPromiseCtor(promiseCtor);
		return new promiseCtor(function(resolve, reject) {
			var value;
			_this.subscribe(function(x) {
				return value = x;
			}, function(err) {
				return reject(err);
			}, function() {
				return resolve(value);
			});
		});
	};
	Observable.create = function(subscribe) {
		return new Observable(subscribe);
	};
	return Observable;
}();
function getPromiseCtor(promiseCtor) {
	var _a;
	return (_a = promiseCtor !== null && promiseCtor !== void 0 ? promiseCtor : config.Promise) !== null && _a !== void 0 ? _a : Promise;
}
function isObserver(value) {
	return value && isFunction(value.next) && isFunction(value.error) && isFunction(value.complete);
}
function isSubscriber(value) {
	return value && value instanceof Subscriber || isObserver(value) && isSubscription(value);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/lift.js
function hasLift(source) {
	return isFunction(source === null || source === void 0 ? void 0 : source.lift);
}
function operate(init) {
	return function(source) {
		if (hasLift(source)) return source.lift(function(liftedSource) {
			try {
				return init(liftedSource, this);
			} catch (err) {
				this.error(err);
			}
		});
		throw new TypeError("Unable to lift unknown Observable type");
	};
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/OperatorSubscriber.js
function createOperatorSubscriber(destination, onNext, onComplete, onError, onFinalize) {
	return new OperatorSubscriber(destination, onNext, onComplete, onError, onFinalize);
}
var OperatorSubscriber = function(_super) {
	__extends(OperatorSubscriber, _super);
	function OperatorSubscriber(destination, onNext, onComplete, onError, onFinalize, shouldUnsubscribe) {
		var _this = _super.call(this, destination) || this;
		_this.onFinalize = onFinalize;
		_this.shouldUnsubscribe = shouldUnsubscribe;
		_this._next = onNext ? function(value) {
			try {
				onNext(value);
			} catch (err) {
				destination.error(err);
			}
		} : _super.prototype._next;
		_this._error = onError ? function(err) {
			try {
				onError(err);
			} catch (err) {
				destination.error(err);
			} finally {
				this.unsubscribe();
			}
		} : _super.prototype._error;
		_this._complete = onComplete ? function() {
			try {
				onComplete();
			} catch (err) {
				destination.error(err);
			} finally {
				this.unsubscribe();
			}
		} : _super.prototype._complete;
		return _this;
	}
	OperatorSubscriber.prototype.unsubscribe = function() {
		var _a;
		if (!this.shouldUnsubscribe || this.shouldUnsubscribe()) {
			var closed_1 = this.closed;
			_super.prototype.unsubscribe.call(this);
			!closed_1 && ((_a = this.onFinalize) === null || _a === void 0 || _a.call(this));
		}
	};
	return OperatorSubscriber;
}(Subscriber);
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/refCount.js
function refCount() {
	return operate(function(source, subscriber) {
		var connection = null;
		source._refCount++;
		var refCounter = createOperatorSubscriber(subscriber, void 0, void 0, void 0, function() {
			if (!source || source._refCount <= 0 || 0 < --source._refCount) {
				connection = null;
				return;
			}
			var sharedConnection = source._connection;
			var conn = connection;
			connection = null;
			if (sharedConnection && (!conn || sharedConnection === conn)) sharedConnection.unsubscribe();
			subscriber.unsubscribe();
		});
		source.subscribe(refCounter);
		if (!refCounter.closed) connection = source.connect();
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/observable/ConnectableObservable.js
var ConnectableObservable = function(_super) {
	__extends(ConnectableObservable, _super);
	function ConnectableObservable(source, subjectFactory) {
		var _this = _super.call(this) || this;
		_this.source = source;
		_this.subjectFactory = subjectFactory;
		_this._subject = null;
		_this._refCount = 0;
		_this._connection = null;
		if (hasLift(source)) _this.lift = source.lift;
		return _this;
	}
	ConnectableObservable.prototype._subscribe = function(subscriber) {
		return this.getSubject().subscribe(subscriber);
	};
	ConnectableObservable.prototype.getSubject = function() {
		var subject = this._subject;
		if (!subject || subject.isStopped) this._subject = this.subjectFactory();
		return this._subject;
	};
	ConnectableObservable.prototype._teardown = function() {
		this._refCount = 0;
		var _connection = this._connection;
		this._subject = this._connection = null;
		_connection === null || _connection === void 0 || _connection.unsubscribe();
	};
	ConnectableObservable.prototype.connect = function() {
		var _this = this;
		var connection = this._connection;
		if (!connection) {
			connection = this._connection = new Subscription();
			var subject_1 = this.getSubject();
			connection.add(this.source.subscribe(createOperatorSubscriber(subject_1, void 0, function() {
				_this._teardown();
				subject_1.complete();
			}, function(err) {
				_this._teardown();
				subject_1.error(err);
			}, function() {
				return _this._teardown();
			})));
			if (connection.closed) {
				this._connection = null;
				connection = Subscription.EMPTY;
			}
		}
		return connection;
	};
	ConnectableObservable.prototype.refCount = function() {
		return refCount()(this);
	};
	return ConnectableObservable;
}(Observable);
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/scheduler/performanceTimestampProvider.js
var performanceTimestampProvider = {
	now: function() {
		return (performanceTimestampProvider.delegate || performance).now();
	},
	delegate: void 0
};
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/scheduler/animationFrameProvider.js
var animationFrameProvider = {
	schedule: function(callback) {
		var request = requestAnimationFrame;
		var cancel = cancelAnimationFrame;
		var delegate = animationFrameProvider.delegate;
		if (delegate) {
			request = delegate.requestAnimationFrame;
			cancel = delegate.cancelAnimationFrame;
		}
		var handle = request(function(timestamp) {
			cancel = void 0;
			callback(timestamp);
		});
		return new Subscription(function() {
			return cancel === null || cancel === void 0 ? void 0 : cancel(handle);
		});
	},
	requestAnimationFrame: function() {
		var args = [];
		for (var _i = 0; _i < arguments.length; _i++) args[_i] = arguments[_i];
		var delegate = animationFrameProvider.delegate;
		return ((delegate === null || delegate === void 0 ? void 0 : delegate.requestAnimationFrame) || requestAnimationFrame).apply(void 0, __spreadArray([], __read(args)));
	},
	cancelAnimationFrame: function() {
		var args = [];
		for (var _i = 0; _i < arguments.length; _i++) args[_i] = arguments[_i];
		var delegate = animationFrameProvider.delegate;
		return ((delegate === null || delegate === void 0 ? void 0 : delegate.cancelAnimationFrame) || cancelAnimationFrame).apply(void 0, __spreadArray([], __read(args)));
	},
	delegate: void 0
};
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/observable/dom/animationFrames.js
function animationFrames(timestampProvider) {
	return timestampProvider ? animationFramesFactory(timestampProvider) : DEFAULT_ANIMATION_FRAMES;
}
function animationFramesFactory(timestampProvider) {
	return new Observable(function(subscriber) {
		var provider = timestampProvider || performanceTimestampProvider;
		var start = provider.now();
		var id = 0;
		var run = function() {
			if (!subscriber.closed) id = animationFrameProvider.requestAnimationFrame(function(timestamp) {
				id = 0;
				var now = provider.now();
				subscriber.next({
					timestamp: timestampProvider ? now : timestamp,
					elapsed: now - start
				});
				run();
			});
		};
		run();
		return function() {
			if (id) animationFrameProvider.cancelAnimationFrame(id);
		};
	});
}
var DEFAULT_ANIMATION_FRAMES = animationFramesFactory();
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/ObjectUnsubscribedError.js
var ObjectUnsubscribedError = createErrorClass(function(_super) {
	return function ObjectUnsubscribedErrorImpl() {
		_super(this);
		this.name = "ObjectUnsubscribedError";
		this.message = "object unsubscribed";
	};
});
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/Subject.js
var Subject = function(_super) {
	__extends(Subject, _super);
	function Subject() {
		var _this = _super.call(this) || this;
		_this.closed = false;
		_this.currentObservers = null;
		_this.observers = [];
		_this.isStopped = false;
		_this.hasError = false;
		_this.thrownError = null;
		return _this;
	}
	Subject.prototype.lift = function(operator) {
		var subject = new AnonymousSubject(this, this);
		subject.operator = operator;
		return subject;
	};
	Subject.prototype._throwIfClosed = function() {
		if (this.closed) throw new ObjectUnsubscribedError();
	};
	Subject.prototype.next = function(value) {
		var _this = this;
		errorContext(function() {
			var e_1, _a;
			_this._throwIfClosed();
			if (!_this.isStopped) {
				if (!_this.currentObservers) _this.currentObservers = Array.from(_this.observers);
				try {
					for (var _b = __values(_this.currentObservers), _c = _b.next(); !_c.done; _c = _b.next()) _c.value.next(value);
				} catch (e_1_1) {
					e_1 = { error: e_1_1 };
				} finally {
					try {
						if (_c && !_c.done && (_a = _b.return)) _a.call(_b);
					} finally {
						if (e_1) throw e_1.error;
					}
				}
			}
		});
	};
	Subject.prototype.error = function(err) {
		var _this = this;
		errorContext(function() {
			_this._throwIfClosed();
			if (!_this.isStopped) {
				_this.hasError = _this.isStopped = true;
				_this.thrownError = err;
				var observers = _this.observers;
				while (observers.length) observers.shift().error(err);
			}
		});
	};
	Subject.prototype.complete = function() {
		var _this = this;
		errorContext(function() {
			_this._throwIfClosed();
			if (!_this.isStopped) {
				_this.isStopped = true;
				var observers = _this.observers;
				while (observers.length) observers.shift().complete();
			}
		});
	};
	Subject.prototype.unsubscribe = function() {
		this.isStopped = this.closed = true;
		this.observers = this.currentObservers = null;
	};
	Object.defineProperty(Subject.prototype, "observed", {
		get: function() {
			var _a;
			return ((_a = this.observers) === null || _a === void 0 ? void 0 : _a.length) > 0;
		},
		enumerable: false,
		configurable: true
	});
	Subject.prototype._trySubscribe = function(subscriber) {
		this._throwIfClosed();
		return _super.prototype._trySubscribe.call(this, subscriber);
	};
	Subject.prototype._subscribe = function(subscriber) {
		this._throwIfClosed();
		this._checkFinalizedStatuses(subscriber);
		return this._innerSubscribe(subscriber);
	};
	Subject.prototype._innerSubscribe = function(subscriber) {
		var _this = this;
		var _a = this, hasError = _a.hasError, isStopped = _a.isStopped, observers = _a.observers;
		if (hasError || isStopped) return EMPTY_SUBSCRIPTION;
		this.currentObservers = null;
		observers.push(subscriber);
		return new Subscription(function() {
			_this.currentObservers = null;
			arrRemove(observers, subscriber);
		});
	};
	Subject.prototype._checkFinalizedStatuses = function(subscriber) {
		var _a = this, hasError = _a.hasError, thrownError = _a.thrownError, isStopped = _a.isStopped;
		if (hasError) subscriber.error(thrownError);
		else if (isStopped) subscriber.complete();
	};
	Subject.prototype.asObservable = function() {
		var observable = new Observable();
		observable.source = this;
		return observable;
	};
	Subject.create = function(destination, source) {
		return new AnonymousSubject(destination, source);
	};
	return Subject;
}(Observable);
var AnonymousSubject = function(_super) {
	__extends(AnonymousSubject, _super);
	function AnonymousSubject(destination, source) {
		var _this = _super.call(this) || this;
		_this.destination = destination;
		_this.source = source;
		return _this;
	}
	AnonymousSubject.prototype.next = function(value) {
		var _a, _b;
		(_b = (_a = this.destination) === null || _a === void 0 ? void 0 : _a.next) === null || _b === void 0 || _b.call(_a, value);
	};
	AnonymousSubject.prototype.error = function(err) {
		var _a, _b;
		(_b = (_a = this.destination) === null || _a === void 0 ? void 0 : _a.error) === null || _b === void 0 || _b.call(_a, err);
	};
	AnonymousSubject.prototype.complete = function() {
		var _a, _b;
		(_b = (_a = this.destination) === null || _a === void 0 ? void 0 : _a.complete) === null || _b === void 0 || _b.call(_a);
	};
	AnonymousSubject.prototype._subscribe = function(subscriber) {
		var _a, _b;
		return (_b = (_a = this.source) === null || _a === void 0 ? void 0 : _a.subscribe(subscriber)) !== null && _b !== void 0 ? _b : EMPTY_SUBSCRIPTION;
	};
	return AnonymousSubject;
}(Subject);
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/BehaviorSubject.js
var BehaviorSubject = function(_super) {
	__extends(BehaviorSubject, _super);
	function BehaviorSubject(_value) {
		var _this = _super.call(this) || this;
		_this._value = _value;
		return _this;
	}
	Object.defineProperty(BehaviorSubject.prototype, "value", {
		get: function() {
			return this.getValue();
		},
		enumerable: false,
		configurable: true
	});
	BehaviorSubject.prototype._subscribe = function(subscriber) {
		var subscription = _super.prototype._subscribe.call(this, subscriber);
		!subscription.closed && subscriber.next(this._value);
		return subscription;
	};
	BehaviorSubject.prototype.getValue = function() {
		var _a = this, hasError = _a.hasError, thrownError = _a.thrownError, _value = _a._value;
		if (hasError) throw thrownError;
		this._throwIfClosed();
		return _value;
	};
	BehaviorSubject.prototype.next = function(value) {
		_super.prototype.next.call(this, this._value = value);
	};
	return BehaviorSubject;
}(Subject);
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/scheduler/dateTimestampProvider.js
var dateTimestampProvider = {
	now: function() {
		return (dateTimestampProvider.delegate || Date).now();
	},
	delegate: void 0
};
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/ReplaySubject.js
var ReplaySubject = function(_super) {
	__extends(ReplaySubject, _super);
	function ReplaySubject(_bufferSize, _windowTime, _timestampProvider) {
		if (_bufferSize === void 0) _bufferSize = Infinity;
		if (_windowTime === void 0) _windowTime = Infinity;
		if (_timestampProvider === void 0) _timestampProvider = dateTimestampProvider;
		var _this = _super.call(this) || this;
		_this._bufferSize = _bufferSize;
		_this._windowTime = _windowTime;
		_this._timestampProvider = _timestampProvider;
		_this._buffer = [];
		_this._infiniteTimeWindow = true;
		_this._infiniteTimeWindow = _windowTime === Infinity;
		_this._bufferSize = Math.max(1, _bufferSize);
		_this._windowTime = Math.max(1, _windowTime);
		return _this;
	}
	ReplaySubject.prototype.next = function(value) {
		var _a = this, isStopped = _a.isStopped, _buffer = _a._buffer, _infiniteTimeWindow = _a._infiniteTimeWindow, _timestampProvider = _a._timestampProvider, _windowTime = _a._windowTime;
		if (!isStopped) {
			_buffer.push(value);
			!_infiniteTimeWindow && _buffer.push(_timestampProvider.now() + _windowTime);
		}
		this._trimBuffer();
		_super.prototype.next.call(this, value);
	};
	ReplaySubject.prototype._subscribe = function(subscriber) {
		this._throwIfClosed();
		this._trimBuffer();
		var subscription = this._innerSubscribe(subscriber);
		var _a = this, _infiniteTimeWindow = _a._infiniteTimeWindow;
		var copy = _a._buffer.slice();
		for (var i = 0; i < copy.length && !subscriber.closed; i += _infiniteTimeWindow ? 1 : 2) subscriber.next(copy[i]);
		this._checkFinalizedStatuses(subscriber);
		return subscription;
	};
	ReplaySubject.prototype._trimBuffer = function() {
		var _a = this, _bufferSize = _a._bufferSize, _timestampProvider = _a._timestampProvider, _buffer = _a._buffer, _infiniteTimeWindow = _a._infiniteTimeWindow;
		var adjustedBufferSize = (_infiniteTimeWindow ? 1 : 2) * _bufferSize;
		_bufferSize < Infinity && adjustedBufferSize < _buffer.length && _buffer.splice(0, _buffer.length - adjustedBufferSize);
		if (!_infiniteTimeWindow) {
			var now = _timestampProvider.now();
			var last = 0;
			for (var i = 1; i < _buffer.length && _buffer[i] <= now; i += 2) last = i;
			last && _buffer.splice(0, last + 1);
		}
	};
	return ReplaySubject;
}(Subject);
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/AsyncSubject.js
var AsyncSubject = function(_super) {
	__extends(AsyncSubject, _super);
	function AsyncSubject() {
		var _this = _super !== null && _super.apply(this, arguments) || this;
		_this._value = null;
		_this._hasValue = false;
		_this._isComplete = false;
		return _this;
	}
	AsyncSubject.prototype._checkFinalizedStatuses = function(subscriber) {
		var _a = this, hasError = _a.hasError, _hasValue = _a._hasValue, _value = _a._value, thrownError = _a.thrownError, isStopped = _a.isStopped, _isComplete = _a._isComplete;
		if (hasError) subscriber.error(thrownError);
		else if (isStopped || _isComplete) {
			_hasValue && subscriber.next(_value);
			subscriber.complete();
		}
	};
	AsyncSubject.prototype.next = function(value) {
		if (!this.isStopped) {
			this._value = value;
			this._hasValue = true;
		}
	};
	AsyncSubject.prototype.complete = function() {
		var _a = this, _hasValue = _a._hasValue, _value = _a._value;
		if (!_a._isComplete) {
			this._isComplete = true;
			_hasValue && _super.prototype.next.call(this, _value);
			_super.prototype.complete.call(this);
		}
	};
	return AsyncSubject;
}(Subject);
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/scheduler/Action.js
var Action = function(_super) {
	__extends(Action, _super);
	function Action(scheduler, work) {
		return _super.call(this) || this;
	}
	Action.prototype.schedule = function(state, delay) {
		if (delay === void 0) delay = 0;
		return this;
	};
	return Action;
}(Subscription);
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/scheduler/intervalProvider.js
var intervalProvider = {
	setInterval: function(handler, timeout) {
		var args = [];
		for (var _i = 2; _i < arguments.length; _i++) args[_i - 2] = arguments[_i];
		var delegate = intervalProvider.delegate;
		if (delegate === null || delegate === void 0 ? void 0 : delegate.setInterval) return delegate.setInterval.apply(delegate, __spreadArray([handler, timeout], __read(args)));
		return setInterval.apply(void 0, __spreadArray([handler, timeout], __read(args)));
	},
	clearInterval: function(handle) {
		var delegate = intervalProvider.delegate;
		return ((delegate === null || delegate === void 0 ? void 0 : delegate.clearInterval) || clearInterval)(handle);
	},
	delegate: void 0
};
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/scheduler/AsyncAction.js
var AsyncAction = function(_super) {
	__extends(AsyncAction, _super);
	function AsyncAction(scheduler, work) {
		var _this = _super.call(this, scheduler, work) || this;
		_this.scheduler = scheduler;
		_this.work = work;
		_this.pending = false;
		return _this;
	}
	AsyncAction.prototype.schedule = function(state, delay) {
		var _a;
		if (delay === void 0) delay = 0;
		if (this.closed) return this;
		this.state = state;
		var id = this.id;
		var scheduler = this.scheduler;
		if (id != null) this.id = this.recycleAsyncId(scheduler, id, delay);
		this.pending = true;
		this.delay = delay;
		this.id = (_a = this.id) !== null && _a !== void 0 ? _a : this.requestAsyncId(scheduler, this.id, delay);
		return this;
	};
	AsyncAction.prototype.requestAsyncId = function(scheduler, _id, delay) {
		if (delay === void 0) delay = 0;
		return intervalProvider.setInterval(scheduler.flush.bind(scheduler, this), delay);
	};
	AsyncAction.prototype.recycleAsyncId = function(_scheduler, id, delay) {
		if (delay === void 0) delay = 0;
		if (delay != null && this.delay === delay && this.pending === false) return id;
		if (id != null) intervalProvider.clearInterval(id);
	};
	AsyncAction.prototype.execute = function(state, delay) {
		if (this.closed) return /* @__PURE__ */ new Error("executing a cancelled action");
		this.pending = false;
		var error = this._execute(state, delay);
		if (error) return error;
		else if (this.pending === false && this.id != null) this.id = this.recycleAsyncId(this.scheduler, this.id, null);
	};
	AsyncAction.prototype._execute = function(state, _delay) {
		var errored = false;
		var errorValue;
		try {
			this.work(state);
		} catch (e) {
			errored = true;
			errorValue = e ? e : /* @__PURE__ */ new Error("Scheduled action threw falsy error");
		}
		if (errored) {
			this.unsubscribe();
			return errorValue;
		}
	};
	AsyncAction.prototype.unsubscribe = function() {
		if (!this.closed) {
			var _a = this, id = _a.id, scheduler = _a.scheduler;
			var actions = scheduler.actions;
			this.work = this.state = this.scheduler = null;
			this.pending = false;
			arrRemove(actions, this);
			if (id != null) this.id = this.recycleAsyncId(scheduler, id, null);
			this.delay = null;
			_super.prototype.unsubscribe.call(this);
		}
	};
	return AsyncAction;
}(Action);
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/Immediate.js
var nextHandle = 1;
var resolved;
var activeHandles = {};
function findAndClearHandle(handle) {
	if (handle in activeHandles) {
		delete activeHandles[handle];
		return true;
	}
	return false;
}
var Immediate = {
	setImmediate: function(cb) {
		var handle = nextHandle++;
		activeHandles[handle] = true;
		if (!resolved) resolved = Promise.resolve();
		resolved.then(function() {
			return findAndClearHandle(handle) && cb();
		});
		return handle;
	},
	clearImmediate: function(handle) {
		findAndClearHandle(handle);
	}
};
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/scheduler/immediateProvider.js
var setImmediate = Immediate.setImmediate;
var clearImmediate = Immediate.clearImmediate;
var immediateProvider = {
	setImmediate: function() {
		var args = [];
		for (var _i = 0; _i < arguments.length; _i++) args[_i] = arguments[_i];
		var delegate = immediateProvider.delegate;
		return ((delegate === null || delegate === void 0 ? void 0 : delegate.setImmediate) || setImmediate).apply(void 0, __spreadArray([], __read(args)));
	},
	clearImmediate: function(handle) {
		var delegate = immediateProvider.delegate;
		return ((delegate === null || delegate === void 0 ? void 0 : delegate.clearImmediate) || clearImmediate)(handle);
	},
	delegate: void 0
};
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/scheduler/AsapAction.js
var AsapAction = function(_super) {
	__extends(AsapAction, _super);
	function AsapAction(scheduler, work) {
		var _this = _super.call(this, scheduler, work) || this;
		_this.scheduler = scheduler;
		_this.work = work;
		return _this;
	}
	AsapAction.prototype.requestAsyncId = function(scheduler, id, delay) {
		if (delay === void 0) delay = 0;
		if (delay !== null && delay > 0) return _super.prototype.requestAsyncId.call(this, scheduler, id, delay);
		scheduler.actions.push(this);
		return scheduler._scheduled || (scheduler._scheduled = immediateProvider.setImmediate(scheduler.flush.bind(scheduler, void 0)));
	};
	AsapAction.prototype.recycleAsyncId = function(scheduler, id, delay) {
		var _a;
		if (delay === void 0) delay = 0;
		if (delay != null ? delay > 0 : this.delay > 0) return _super.prototype.recycleAsyncId.call(this, scheduler, id, delay);
		var actions = scheduler.actions;
		if (id != null && ((_a = actions[actions.length - 1]) === null || _a === void 0 ? void 0 : _a.id) !== id) {
			immediateProvider.clearImmediate(id);
			if (scheduler._scheduled === id) scheduler._scheduled = void 0;
		}
	};
	return AsapAction;
}(AsyncAction);
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/Scheduler.js
var Scheduler = function() {
	function Scheduler(schedulerActionCtor, now) {
		if (now === void 0) now = Scheduler.now;
		this.schedulerActionCtor = schedulerActionCtor;
		this.now = now;
	}
	Scheduler.prototype.schedule = function(work, delay, state) {
		if (delay === void 0) delay = 0;
		return new this.schedulerActionCtor(this, work).schedule(state, delay);
	};
	Scheduler.now = dateTimestampProvider.now;
	return Scheduler;
}();
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/scheduler/AsyncScheduler.js
var AsyncScheduler = function(_super) {
	__extends(AsyncScheduler, _super);
	function AsyncScheduler(SchedulerAction, now) {
		if (now === void 0) now = Scheduler.now;
		var _this = _super.call(this, SchedulerAction, now) || this;
		_this.actions = [];
		_this._active = false;
		return _this;
	}
	AsyncScheduler.prototype.flush = function(action) {
		var actions = this.actions;
		if (this._active) {
			actions.push(action);
			return;
		}
		var error;
		this._active = true;
		do
			if (error = action.execute(action.state, action.delay)) break;
		while (action = actions.shift());
		this._active = false;
		if (error) {
			while (action = actions.shift()) action.unsubscribe();
			throw error;
		}
	};
	return AsyncScheduler;
}(Scheduler);
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/scheduler/asap.js
var asapScheduler = new (function(_super) {
	__extends(AsapScheduler, _super);
	function AsapScheduler() {
		return _super !== null && _super.apply(this, arguments) || this;
	}
	AsapScheduler.prototype.flush = function(action) {
		this._active = true;
		var flushId = this._scheduled;
		this._scheduled = void 0;
		var actions = this.actions;
		var error;
		action = action || actions.shift();
		do
			if (error = action.execute(action.state, action.delay)) break;
		while ((action = actions[0]) && action.id === flushId && actions.shift());
		this._active = false;
		if (error) {
			while ((action = actions[0]) && action.id === flushId && actions.shift()) action.unsubscribe();
			throw error;
		}
	};
	return AsapScheduler;
}(AsyncScheduler))(AsapAction);
var asap = asapScheduler;
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/scheduler/async.js
var asyncScheduler = new AsyncScheduler(AsyncAction);
var async = asyncScheduler;
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/scheduler/QueueAction.js
var QueueAction = function(_super) {
	__extends(QueueAction, _super);
	function QueueAction(scheduler, work) {
		var _this = _super.call(this, scheduler, work) || this;
		_this.scheduler = scheduler;
		_this.work = work;
		return _this;
	}
	QueueAction.prototype.schedule = function(state, delay) {
		if (delay === void 0) delay = 0;
		if (delay > 0) return _super.prototype.schedule.call(this, state, delay);
		this.delay = delay;
		this.state = state;
		this.scheduler.flush(this);
		return this;
	};
	QueueAction.prototype.execute = function(state, delay) {
		return delay > 0 || this.closed ? _super.prototype.execute.call(this, state, delay) : this._execute(state, delay);
	};
	QueueAction.prototype.requestAsyncId = function(scheduler, id, delay) {
		if (delay === void 0) delay = 0;
		if (delay != null && delay > 0 || delay == null && this.delay > 0) return _super.prototype.requestAsyncId.call(this, scheduler, id, delay);
		scheduler.flush(this);
		return 0;
	};
	return QueueAction;
}(AsyncAction);
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/scheduler/queue.js
var queueScheduler = new (function(_super) {
	__extends(QueueScheduler, _super);
	function QueueScheduler() {
		return _super !== null && _super.apply(this, arguments) || this;
	}
	return QueueScheduler;
}(AsyncScheduler))(QueueAction);
var queue = queueScheduler;
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/scheduler/AnimationFrameAction.js
var AnimationFrameAction = function(_super) {
	__extends(AnimationFrameAction, _super);
	function AnimationFrameAction(scheduler, work) {
		var _this = _super.call(this, scheduler, work) || this;
		_this.scheduler = scheduler;
		_this.work = work;
		return _this;
	}
	AnimationFrameAction.prototype.requestAsyncId = function(scheduler, id, delay) {
		if (delay === void 0) delay = 0;
		if (delay !== null && delay > 0) return _super.prototype.requestAsyncId.call(this, scheduler, id, delay);
		scheduler.actions.push(this);
		return scheduler._scheduled || (scheduler._scheduled = animationFrameProvider.requestAnimationFrame(function() {
			return scheduler.flush(void 0);
		}));
	};
	AnimationFrameAction.prototype.recycleAsyncId = function(scheduler, id, delay) {
		var _a;
		if (delay === void 0) delay = 0;
		if (delay != null ? delay > 0 : this.delay > 0) return _super.prototype.recycleAsyncId.call(this, scheduler, id, delay);
		var actions = scheduler.actions;
		if (id != null && id === scheduler._scheduled && ((_a = actions[actions.length - 1]) === null || _a === void 0 ? void 0 : _a.id) !== id) {
			animationFrameProvider.cancelAnimationFrame(id);
			scheduler._scheduled = void 0;
		}
	};
	return AnimationFrameAction;
}(AsyncAction);
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/scheduler/animationFrame.js
var animationFrameScheduler = new (function(_super) {
	__extends(AnimationFrameScheduler, _super);
	function AnimationFrameScheduler() {
		return _super !== null && _super.apply(this, arguments) || this;
	}
	AnimationFrameScheduler.prototype.flush = function(action) {
		this._active = true;
		var flushId;
		if (action) flushId = action.id;
		else {
			flushId = this._scheduled;
			this._scheduled = void 0;
		}
		var actions = this.actions;
		var error;
		action = action || actions.shift();
		do
			if (error = action.execute(action.state, action.delay)) break;
		while ((action = actions[0]) && action.id === flushId && actions.shift());
		this._active = false;
		if (error) {
			while ((action = actions[0]) && action.id === flushId && actions.shift()) action.unsubscribe();
			throw error;
		}
	};
	return AnimationFrameScheduler;
}(AsyncScheduler))(AnimationFrameAction);
var animationFrame = animationFrameScheduler;
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/scheduler/VirtualTimeScheduler.js
var VirtualTimeScheduler = function(_super) {
	__extends(VirtualTimeScheduler, _super);
	function VirtualTimeScheduler(schedulerActionCtor, maxFrames) {
		if (schedulerActionCtor === void 0) schedulerActionCtor = VirtualAction;
		if (maxFrames === void 0) maxFrames = Infinity;
		var _this = _super.call(this, schedulerActionCtor, function() {
			return _this.frame;
		}) || this;
		_this.maxFrames = maxFrames;
		_this.frame = 0;
		_this.index = -1;
		return _this;
	}
	VirtualTimeScheduler.prototype.flush = function() {
		var _a = this, actions = _a.actions, maxFrames = _a.maxFrames;
		var error;
		var action;
		while ((action = actions[0]) && action.delay <= maxFrames) {
			actions.shift();
			this.frame = action.delay;
			if (error = action.execute(action.state, action.delay)) break;
		}
		if (error) {
			while (action = actions.shift()) action.unsubscribe();
			throw error;
		}
	};
	VirtualTimeScheduler.frameTimeFactor = 10;
	return VirtualTimeScheduler;
}(AsyncScheduler);
var VirtualAction = function(_super) {
	__extends(VirtualAction, _super);
	function VirtualAction(scheduler, work, index) {
		if (index === void 0) index = scheduler.index += 1;
		var _this = _super.call(this, scheduler, work) || this;
		_this.scheduler = scheduler;
		_this.work = work;
		_this.index = index;
		_this.active = true;
		_this.index = scheduler.index = index;
		return _this;
	}
	VirtualAction.prototype.schedule = function(state, delay) {
		if (delay === void 0) delay = 0;
		if (Number.isFinite(delay)) {
			if (!this.id) return _super.prototype.schedule.call(this, state, delay);
			this.active = false;
			var action = new VirtualAction(this.scheduler, this.work);
			this.add(action);
			return action.schedule(state, delay);
		} else return Subscription.EMPTY;
	};
	VirtualAction.prototype.requestAsyncId = function(scheduler, id, delay) {
		if (delay === void 0) delay = 0;
		this.delay = scheduler.frame + delay;
		var actions = scheduler.actions;
		actions.push(this);
		actions.sort(VirtualAction.sortActions);
		return 1;
	};
	VirtualAction.prototype.recycleAsyncId = function(scheduler, id, delay) {
		if (delay === void 0) delay = 0;
	};
	VirtualAction.prototype._execute = function(state, delay) {
		if (this.active === true) return _super.prototype._execute.call(this, state, delay);
	};
	VirtualAction.sortActions = function(a, b) {
		if (a.delay === b.delay) if (a.index === b.index) return 0;
		else if (a.index > b.index) return 1;
		else return -1;
		else if (a.delay > b.delay) return 1;
		else return -1;
	};
	return VirtualAction;
}(AsyncAction);
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/observable/empty.js
var EMPTY = new Observable(function(subscriber) {
	return subscriber.complete();
});
function empty(scheduler) {
	return scheduler ? emptyScheduled(scheduler) : EMPTY;
}
function emptyScheduled(scheduler) {
	return new Observable(function(subscriber) {
		return scheduler.schedule(function() {
			return subscriber.complete();
		});
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/isScheduler.js
function isScheduler(value) {
	return value && isFunction(value.schedule);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/args.js
function last$1(arr) {
	return arr[arr.length - 1];
}
function popResultSelector(args) {
	return isFunction(last$1(args)) ? args.pop() : void 0;
}
function popScheduler(args) {
	return isScheduler(last$1(args)) ? args.pop() : void 0;
}
function popNumber(args, defaultValue) {
	return typeof last$1(args) === "number" ? args.pop() : defaultValue;
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/isArrayLike.js
var isArrayLike = (function(x) {
	return x && typeof x.length === "number" && typeof x !== "function";
});
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/isPromise.js
function isPromise(value) {
	return isFunction(value === null || value === void 0 ? void 0 : value.then);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/isInteropObservable.js
function isInteropObservable(input) {
	return isFunction(input[observable]);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/isAsyncIterable.js
function isAsyncIterable(obj) {
	return Symbol.asyncIterator && isFunction(obj === null || obj === void 0 ? void 0 : obj[Symbol.asyncIterator]);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/throwUnobservableError.js
function createInvalidObservableTypeError(input) {
	return /* @__PURE__ */ new TypeError("You provided " + (input !== null && typeof input === "object" ? "an invalid object" : "'" + input + "'") + " where a stream was expected. You can provide an Observable, Promise, ReadableStream, Array, AsyncIterable, or Iterable.");
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/symbol/iterator.js
function getSymbolIterator() {
	if (typeof Symbol !== "function" || !Symbol.iterator) return "@@iterator";
	return Symbol.iterator;
}
var iterator = getSymbolIterator();
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/isIterable.js
function isIterable(input) {
	return isFunction(input === null || input === void 0 ? void 0 : input[iterator]);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/isReadableStreamLike.js
function readableStreamLikeToAsyncGenerator(readableStream) {
	return __asyncGenerator(this, arguments, function readableStreamLikeToAsyncGenerator_1() {
		var reader, _a, value, done;
		return __generator(this, function(_b) {
			switch (_b.label) {
				case 0:
					reader = readableStream.getReader();
					_b.label = 1;
				case 1:
					_b.trys.push([
						1,
						,
						9,
						10
					]);
					_b.label = 2;
				case 2: return [4, __await(reader.read())];
				case 3:
					_a = _b.sent(), value = _a.value, done = _a.done;
					if (!done) return [3, 5];
					return [4, __await(void 0)];
				case 4: return [2, _b.sent()];
				case 5: return [4, __await(value)];
				case 6: return [4, _b.sent()];
				case 7:
					_b.sent();
					return [3, 2];
				case 8: return [3, 10];
				case 9:
					reader.releaseLock();
					return [7];
				case 10: return [2];
			}
		});
	});
}
function isReadableStreamLike(obj) {
	return isFunction(obj === null || obj === void 0 ? void 0 : obj.getReader);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/observable/innerFrom.js
function innerFrom(input) {
	if (input instanceof Observable) return input;
	if (input != null) {
		if (isInteropObservable(input)) return fromInteropObservable(input);
		if (isArrayLike(input)) return fromArrayLike(input);
		if (isPromise(input)) return fromPromise(input);
		if (isAsyncIterable(input)) return fromAsyncIterable(input);
		if (isIterable(input)) return fromIterable(input);
		if (isReadableStreamLike(input)) return fromReadableStreamLike(input);
	}
	throw createInvalidObservableTypeError(input);
}
function fromInteropObservable(obj) {
	return new Observable(function(subscriber) {
		var obs = obj[observable]();
		if (isFunction(obs.subscribe)) return obs.subscribe(subscriber);
		throw new TypeError("Provided object does not correctly implement Symbol.observable");
	});
}
function fromArrayLike(array) {
	return new Observable(function(subscriber) {
		for (var i = 0; i < array.length && !subscriber.closed; i++) subscriber.next(array[i]);
		subscriber.complete();
	});
}
function fromPromise(promise) {
	return new Observable(function(subscriber) {
		promise.then(function(value) {
			if (!subscriber.closed) {
				subscriber.next(value);
				subscriber.complete();
			}
		}, function(err) {
			return subscriber.error(err);
		}).then(null, reportUnhandledError);
	});
}
function fromIterable(iterable) {
	return new Observable(function(subscriber) {
		var e_1, _a;
		try {
			for (var iterable_1 = __values(iterable), iterable_1_1 = iterable_1.next(); !iterable_1_1.done; iterable_1_1 = iterable_1.next()) {
				var value = iterable_1_1.value;
				subscriber.next(value);
				if (subscriber.closed) return;
			}
		} catch (e_1_1) {
			e_1 = { error: e_1_1 };
		} finally {
			try {
				if (iterable_1_1 && !iterable_1_1.done && (_a = iterable_1.return)) _a.call(iterable_1);
			} finally {
				if (e_1) throw e_1.error;
			}
		}
		subscriber.complete();
	});
}
function fromAsyncIterable(asyncIterable) {
	return new Observable(function(subscriber) {
		process(asyncIterable, subscriber).catch(function(err) {
			return subscriber.error(err);
		});
	});
}
function fromReadableStreamLike(readableStream) {
	return fromAsyncIterable(readableStreamLikeToAsyncGenerator(readableStream));
}
function process(asyncIterable, subscriber) {
	var asyncIterable_1, asyncIterable_1_1;
	var e_2, _a;
	return __awaiter(this, void 0, void 0, function() {
		var value, e_2_1;
		return __generator(this, function(_b) {
			switch (_b.label) {
				case 0:
					_b.trys.push([
						0,
						5,
						6,
						11
					]);
					asyncIterable_1 = __asyncValues(asyncIterable);
					_b.label = 1;
				case 1: return [4, asyncIterable_1.next()];
				case 2:
					if (!(asyncIterable_1_1 = _b.sent(), !asyncIterable_1_1.done)) return [3, 4];
					value = asyncIterable_1_1.value;
					subscriber.next(value);
					if (subscriber.closed) return [2];
					_b.label = 3;
				case 3: return [3, 1];
				case 4: return [3, 11];
				case 5:
					e_2_1 = _b.sent();
					e_2 = { error: e_2_1 };
					return [3, 11];
				case 6:
					_b.trys.push([
						6,
						,
						9,
						10
					]);
					if (!(asyncIterable_1_1 && !asyncIterable_1_1.done && (_a = asyncIterable_1.return))) return [3, 8];
					return [4, _a.call(asyncIterable_1)];
				case 7:
					_b.sent();
					_b.label = 8;
				case 8: return [3, 10];
				case 9:
					if (e_2) throw e_2.error;
					return [7];
				case 10: return [7];
				case 11:
					subscriber.complete();
					return [2];
			}
		});
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/executeSchedule.js
function executeSchedule(parentSubscription, scheduler, work, delay, repeat) {
	if (delay === void 0) delay = 0;
	if (repeat === void 0) repeat = false;
	var scheduleSubscription = scheduler.schedule(function() {
		work();
		if (repeat) parentSubscription.add(this.schedule(null, delay));
		else this.unsubscribe();
	}, delay);
	parentSubscription.add(scheduleSubscription);
	if (!repeat) return scheduleSubscription;
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/observeOn.js
function observeOn(scheduler, delay) {
	if (delay === void 0) delay = 0;
	return operate(function(source, subscriber) {
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			return executeSchedule(subscriber, scheduler, function() {
				return subscriber.next(value);
			}, delay);
		}, function() {
			return executeSchedule(subscriber, scheduler, function() {
				return subscriber.complete();
			}, delay);
		}, function(err) {
			return executeSchedule(subscriber, scheduler, function() {
				return subscriber.error(err);
			}, delay);
		}));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/subscribeOn.js
function subscribeOn(scheduler, delay) {
	if (delay === void 0) delay = 0;
	return operate(function(source, subscriber) {
		subscriber.add(scheduler.schedule(function() {
			return source.subscribe(subscriber);
		}, delay));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/scheduled/scheduleObservable.js
function scheduleObservable(input, scheduler) {
	return innerFrom(input).pipe(subscribeOn(scheduler), observeOn(scheduler));
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/scheduled/schedulePromise.js
function schedulePromise(input, scheduler) {
	return innerFrom(input).pipe(subscribeOn(scheduler), observeOn(scheduler));
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/scheduled/scheduleArray.js
function scheduleArray(input, scheduler) {
	return new Observable(function(subscriber) {
		var i = 0;
		return scheduler.schedule(function() {
			if (i === input.length) subscriber.complete();
			else {
				subscriber.next(input[i++]);
				if (!subscriber.closed) this.schedule();
			}
		});
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/scheduled/scheduleIterable.js
function scheduleIterable(input, scheduler) {
	return new Observable(function(subscriber) {
		var iterator$1;
		executeSchedule(subscriber, scheduler, function() {
			iterator$1 = input[iterator]();
			executeSchedule(subscriber, scheduler, function() {
				var _a;
				var value;
				var done;
				try {
					_a = iterator$1.next(), value = _a.value, done = _a.done;
				} catch (err) {
					subscriber.error(err);
					return;
				}
				if (done) subscriber.complete();
				else subscriber.next(value);
			}, 0, true);
		});
		return function() {
			return isFunction(iterator$1 === null || iterator$1 === void 0 ? void 0 : iterator$1.return) && iterator$1.return();
		};
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/scheduled/scheduleAsyncIterable.js
function scheduleAsyncIterable(input, scheduler) {
	if (!input) throw new Error("Iterable cannot be null");
	return new Observable(function(subscriber) {
		executeSchedule(subscriber, scheduler, function() {
			var iterator = input[Symbol.asyncIterator]();
			executeSchedule(subscriber, scheduler, function() {
				iterator.next().then(function(result) {
					if (result.done) subscriber.complete();
					else subscriber.next(result.value);
				});
			}, 0, true);
		});
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/scheduled/scheduleReadableStreamLike.js
function scheduleReadableStreamLike(input, scheduler) {
	return scheduleAsyncIterable(readableStreamLikeToAsyncGenerator(input), scheduler);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/scheduled/scheduled.js
function scheduled(input, scheduler) {
	if (input != null) {
		if (isInteropObservable(input)) return scheduleObservable(input, scheduler);
		if (isArrayLike(input)) return scheduleArray(input, scheduler);
		if (isPromise(input)) return schedulePromise(input, scheduler);
		if (isAsyncIterable(input)) return scheduleAsyncIterable(input, scheduler);
		if (isIterable(input)) return scheduleIterable(input, scheduler);
		if (isReadableStreamLike(input)) return scheduleReadableStreamLike(input, scheduler);
	}
	throw createInvalidObservableTypeError(input);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/observable/from.js
function from(input, scheduler) {
	return scheduler ? scheduled(input, scheduler) : innerFrom(input);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/observable/of.js
function of() {
	var args = [];
	for (var _i = 0; _i < arguments.length; _i++) args[_i] = arguments[_i];
	return from(args, popScheduler(args));
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/observable/throwError.js
function throwError(errorOrErrorFactory, scheduler) {
	var errorFactory = isFunction(errorOrErrorFactory) ? errorOrErrorFactory : function() {
		return errorOrErrorFactory;
	};
	var init = function(subscriber) {
		return subscriber.error(errorFactory());
	};
	return new Observable(scheduler ? function(subscriber) {
		return scheduler.schedule(init, 0, subscriber);
	} : init);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/Notification.js
var NotificationKind;
(function(NotificationKind) {
	NotificationKind["NEXT"] = "N";
	NotificationKind["ERROR"] = "E";
	NotificationKind["COMPLETE"] = "C";
})(NotificationKind || (NotificationKind = {}));
var Notification = function() {
	function Notification(kind, value, error) {
		this.kind = kind;
		this.value = value;
		this.error = error;
		this.hasValue = kind === "N";
	}
	Notification.prototype.observe = function(observer) {
		return observeNotification(this, observer);
	};
	Notification.prototype.do = function(nextHandler, errorHandler, completeHandler) {
		var _a = this, kind = _a.kind, value = _a.value, error = _a.error;
		return kind === "N" ? nextHandler === null || nextHandler === void 0 ? void 0 : nextHandler(value) : kind === "E" ? errorHandler === null || errorHandler === void 0 ? void 0 : errorHandler(error) : completeHandler === null || completeHandler === void 0 ? void 0 : completeHandler();
	};
	Notification.prototype.accept = function(nextOrObserver, error, complete) {
		var _a;
		return isFunction((_a = nextOrObserver) === null || _a === void 0 ? void 0 : _a.next) ? this.observe(nextOrObserver) : this.do(nextOrObserver, error, complete);
	};
	Notification.prototype.toObservable = function() {
		var _a = this, kind = _a.kind, value = _a.value, error = _a.error;
		var result = kind === "N" ? of(value) : kind === "E" ? throwError(function() {
			return error;
		}) : kind === "C" ? EMPTY : 0;
		if (!result) throw new TypeError("Unexpected notification kind " + kind);
		return result;
	};
	Notification.createNext = function(value) {
		return new Notification("N", value);
	};
	Notification.createError = function(err) {
		return new Notification("E", void 0, err);
	};
	Notification.createComplete = function() {
		return Notification.completeNotification;
	};
	Notification.completeNotification = new Notification("C");
	return Notification;
}();
function observeNotification(notification, observer) {
	var _a, _b, _c;
	var _d = notification, kind = _d.kind, value = _d.value, error = _d.error;
	if (typeof kind !== "string") throw new TypeError("Invalid notification, missing \"kind\"");
	kind === "N" ? (_a = observer.next) === null || _a === void 0 || _a.call(observer, value) : kind === "E" ? (_b = observer.error) === null || _b === void 0 || _b.call(observer, error) : (_c = observer.complete) === null || _c === void 0 || _c.call(observer);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/isObservable.js
function isObservable(obj) {
	return !!obj && (obj instanceof Observable || isFunction(obj.lift) && isFunction(obj.subscribe));
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/EmptyError.js
var EmptyError = createErrorClass(function(_super) {
	return function EmptyErrorImpl() {
		_super(this);
		this.name = "EmptyError";
		this.message = "no elements in sequence";
	};
});
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/lastValueFrom.js
function lastValueFrom(source, config) {
	var hasConfig = typeof config === "object";
	return new Promise(function(resolve, reject) {
		var _hasValue = false;
		var _value;
		source.subscribe({
			next: function(value) {
				_value = value;
				_hasValue = true;
			},
			error: reject,
			complete: function() {
				if (_hasValue) resolve(_value);
				else if (hasConfig) resolve(config.defaultValue);
				else reject(new EmptyError());
			}
		});
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/firstValueFrom.js
function firstValueFrom(source, config) {
	var hasConfig = typeof config === "object";
	return new Promise(function(resolve, reject) {
		var subscriber = new SafeSubscriber({
			next: function(value) {
				resolve(value);
				subscriber.unsubscribe();
			},
			error: reject,
			complete: function() {
				if (hasConfig) resolve(config.defaultValue);
				else reject(new EmptyError());
			}
		});
		source.subscribe(subscriber);
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/ArgumentOutOfRangeError.js
var ArgumentOutOfRangeError = createErrorClass(function(_super) {
	return function ArgumentOutOfRangeErrorImpl() {
		_super(this);
		this.name = "ArgumentOutOfRangeError";
		this.message = "argument out of range";
	};
});
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/NotFoundError.js
var NotFoundError = createErrorClass(function(_super) {
	return function NotFoundErrorImpl(message) {
		_super(this);
		this.name = "NotFoundError";
		this.message = message;
	};
});
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/SequenceError.js
var SequenceError = createErrorClass(function(_super) {
	return function SequenceErrorImpl(message) {
		_super(this);
		this.name = "SequenceError";
		this.message = message;
	};
});
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/isDate.js
function isValidDate(value) {
	return value instanceof Date && !isNaN(value);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/timeout.js
var TimeoutError = createErrorClass(function(_super) {
	return function TimeoutErrorImpl(info) {
		if (info === void 0) info = null;
		_super(this);
		this.message = "Timeout has occurred";
		this.name = "TimeoutError";
		this.info = info;
	};
});
function timeout(config, schedulerArg) {
	var _a = isValidDate(config) ? { first: config } : typeof config === "number" ? { each: config } : config, first = _a.first, each = _a.each, _b = _a.with, _with = _b === void 0 ? timeoutErrorFactory : _b, _c = _a.scheduler, scheduler = _c === void 0 ? schedulerArg !== null && schedulerArg !== void 0 ? schedulerArg : asyncScheduler : _c, _d = _a.meta, meta = _d === void 0 ? null : _d;
	if (first == null && each == null) throw new TypeError("No timeout provided.");
	return operate(function(source, subscriber) {
		var originalSourceSubscription;
		var timerSubscription;
		var lastValue = null;
		var seen = 0;
		var startTimer = function(delay) {
			timerSubscription = executeSchedule(subscriber, scheduler, function() {
				try {
					originalSourceSubscription.unsubscribe();
					innerFrom(_with({
						meta,
						lastValue,
						seen
					})).subscribe(subscriber);
				} catch (err) {
					subscriber.error(err);
				}
			}, delay);
		};
		originalSourceSubscription = source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			timerSubscription === null || timerSubscription === void 0 || timerSubscription.unsubscribe();
			seen++;
			subscriber.next(lastValue = value);
			each > 0 && startTimer(each);
		}, void 0, void 0, function() {
			if (!(timerSubscription === null || timerSubscription === void 0 ? void 0 : timerSubscription.closed)) timerSubscription === null || timerSubscription === void 0 || timerSubscription.unsubscribe();
			lastValue = null;
		}));
		!seen && startTimer(first != null ? typeof first === "number" ? first : +first - scheduler.now() : each);
	});
}
function timeoutErrorFactory(info) {
	throw new TimeoutError(info);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/map.js
function map(project, thisArg) {
	return operate(function(source, subscriber) {
		var index = 0;
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			subscriber.next(project.call(thisArg, value, index++));
		}));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/mapOneOrManyArgs.js
var isArray$2 = Array.isArray;
function callOrApply(fn, args) {
	return isArray$2(args) ? fn.apply(void 0, __spreadArray([], __read(args))) : fn(args);
}
function mapOneOrManyArgs(fn) {
	return map(function(args) {
		return callOrApply(fn, args);
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/observable/bindCallbackInternals.js
function bindCallbackInternals(isNodeStyle, callbackFunc, resultSelector, scheduler) {
	if (resultSelector) if (isScheduler(resultSelector)) scheduler = resultSelector;
	else return function() {
		var args = [];
		for (var _i = 0; _i < arguments.length; _i++) args[_i] = arguments[_i];
		return bindCallbackInternals(isNodeStyle, callbackFunc, scheduler).apply(this, args).pipe(mapOneOrManyArgs(resultSelector));
	};
	if (scheduler) return function() {
		var args = [];
		for (var _i = 0; _i < arguments.length; _i++) args[_i] = arguments[_i];
		return bindCallbackInternals(isNodeStyle, callbackFunc).apply(this, args).pipe(subscribeOn(scheduler), observeOn(scheduler));
	};
	return function() {
		var _this = this;
		var args = [];
		for (var _i = 0; _i < arguments.length; _i++) args[_i] = arguments[_i];
		var subject = new AsyncSubject();
		var uninitialized = true;
		return new Observable(function(subscriber) {
			var subs = subject.subscribe(subscriber);
			if (uninitialized) {
				uninitialized = false;
				var isAsync_1 = false;
				var isComplete_1 = false;
				callbackFunc.apply(_this, __spreadArray(__spreadArray([], __read(args)), [function() {
					var results = [];
					for (var _i = 0; _i < arguments.length; _i++) results[_i] = arguments[_i];
					if (isNodeStyle) {
						var err = results.shift();
						if (err != null) {
							subject.error(err);
							return;
						}
					}
					subject.next(1 < results.length ? results : results[0]);
					isComplete_1 = true;
					if (isAsync_1) subject.complete();
				}]));
				if (isComplete_1) subject.complete();
				isAsync_1 = true;
			}
			return subs;
		});
	};
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/observable/bindCallback.js
function bindCallback(callbackFunc, resultSelector, scheduler) {
	return bindCallbackInternals(false, callbackFunc, resultSelector, scheduler);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/observable/bindNodeCallback.js
function bindNodeCallback(callbackFunc, resultSelector, scheduler) {
	return bindCallbackInternals(true, callbackFunc, resultSelector, scheduler);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/argsArgArrayOrObject.js
var isArray$1 = Array.isArray;
var getPrototypeOf = Object.getPrototypeOf;
var objectProto = Object.prototype;
var getKeys = Object.keys;
function argsArgArrayOrObject(args) {
	if (args.length === 1) {
		var first_1 = args[0];
		if (isArray$1(first_1)) return {
			args: first_1,
			keys: null
		};
		if (isPOJO(first_1)) {
			var keys = getKeys(first_1);
			return {
				args: keys.map(function(key) {
					return first_1[key];
				}),
				keys
			};
		}
	}
	return {
		args,
		keys: null
	};
}
function isPOJO(obj) {
	return obj && typeof obj === "object" && getPrototypeOf(obj) === objectProto;
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/createObject.js
function createObject(keys, values) {
	return keys.reduce(function(result, key, i) {
		return result[key] = values[i], result;
	}, {});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/observable/combineLatest.js
function combineLatest$1() {
	var args = [];
	for (var _i = 0; _i < arguments.length; _i++) args[_i] = arguments[_i];
	var scheduler = popScheduler(args);
	var resultSelector = popResultSelector(args);
	var _a = argsArgArrayOrObject(args), observables = _a.args, keys = _a.keys;
	if (observables.length === 0) return from([], scheduler);
	var result = new Observable(combineLatestInit(observables, scheduler, keys ? function(values) {
		return createObject(keys, values);
	} : identity));
	return resultSelector ? result.pipe(mapOneOrManyArgs(resultSelector)) : result;
}
function combineLatestInit(observables, scheduler, valueTransform) {
	if (valueTransform === void 0) valueTransform = identity;
	return function(subscriber) {
		maybeSchedule(scheduler, function() {
			var length = observables.length;
			var values = new Array(length);
			var active = length;
			var remainingFirstValues = length;
			var _loop_1 = function(i) {
				maybeSchedule(scheduler, function() {
					var source = from(observables[i], scheduler);
					var hasFirstValue = false;
					source.subscribe(createOperatorSubscriber(subscriber, function(value) {
						values[i] = value;
						if (!hasFirstValue) {
							hasFirstValue = true;
							remainingFirstValues--;
						}
						if (!remainingFirstValues) subscriber.next(valueTransform(values.slice()));
					}, function() {
						if (!--active) subscriber.complete();
					}));
				}, subscriber);
			};
			for (var i = 0; i < length; i++) _loop_1(i);
		}, subscriber);
	};
}
function maybeSchedule(scheduler, execute, subscription) {
	if (scheduler) executeSchedule(subscription, scheduler, execute);
	else execute();
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/mergeInternals.js
function mergeInternals(source, subscriber, project, concurrent, onBeforeNext, expand, innerSubScheduler, additionalFinalizer) {
	var buffer = [];
	var active = 0;
	var index = 0;
	var isComplete = false;
	var checkComplete = function() {
		if (isComplete && !buffer.length && !active) subscriber.complete();
	};
	var outerNext = function(value) {
		return active < concurrent ? doInnerSub(value) : buffer.push(value);
	};
	var doInnerSub = function(value) {
		expand && subscriber.next(value);
		active++;
		var innerComplete = false;
		innerFrom(project(value, index++)).subscribe(createOperatorSubscriber(subscriber, function(innerValue) {
			onBeforeNext === null || onBeforeNext === void 0 || onBeforeNext(innerValue);
			if (expand) outerNext(innerValue);
			else subscriber.next(innerValue);
		}, function() {
			innerComplete = true;
		}, void 0, function() {
			if (innerComplete) try {
				active--;
				var _loop_1 = function() {
					var bufferedValue = buffer.shift();
					if (innerSubScheduler) executeSchedule(subscriber, innerSubScheduler, function() {
						return doInnerSub(bufferedValue);
					});
					else doInnerSub(bufferedValue);
				};
				while (buffer.length && active < concurrent) _loop_1();
				checkComplete();
			} catch (err) {
				subscriber.error(err);
			}
		}));
	};
	source.subscribe(createOperatorSubscriber(subscriber, outerNext, function() {
		isComplete = true;
		checkComplete();
	}));
	return function() {
		additionalFinalizer === null || additionalFinalizer === void 0 || additionalFinalizer();
	};
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/mergeMap.js
function mergeMap(project, resultSelector, concurrent) {
	if (concurrent === void 0) concurrent = Infinity;
	if (isFunction(resultSelector)) return mergeMap(function(a, i) {
		return map(function(b, ii) {
			return resultSelector(a, b, i, ii);
		})(innerFrom(project(a, i)));
	}, concurrent);
	else if (typeof resultSelector === "number") concurrent = resultSelector;
	return operate(function(source, subscriber) {
		return mergeInternals(source, subscriber, project, concurrent);
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/mergeAll.js
function mergeAll(concurrent) {
	if (concurrent === void 0) concurrent = Infinity;
	return mergeMap(identity, concurrent);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/concatAll.js
function concatAll() {
	return mergeAll(1);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/observable/concat.js
function concat$1() {
	var args = [];
	for (var _i = 0; _i < arguments.length; _i++) args[_i] = arguments[_i];
	return concatAll()(from(args, popScheduler(args)));
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/observable/defer.js
function defer(observableFactory) {
	return new Observable(function(subscriber) {
		innerFrom(observableFactory()).subscribe(subscriber);
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/observable/connectable.js
var DEFAULT_CONFIG$1 = {
	connector: function() {
		return new Subject();
	},
	resetOnDisconnect: true
};
function connectable(source, config) {
	if (config === void 0) config = DEFAULT_CONFIG$1;
	var connection = null;
	var connector = config.connector, _a = config.resetOnDisconnect, resetOnDisconnect = _a === void 0 ? true : _a;
	var subject = connector();
	var result = new Observable(function(subscriber) {
		return subject.subscribe(subscriber);
	});
	result.connect = function() {
		if (!connection || connection.closed) {
			connection = defer(function() {
				return source;
			}).subscribe(subject);
			if (resetOnDisconnect) connection.add(function() {
				return subject = connector();
			});
		}
		return connection;
	};
	return result;
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/observable/forkJoin.js
function forkJoin() {
	var args = [];
	for (var _i = 0; _i < arguments.length; _i++) args[_i] = arguments[_i];
	var resultSelector = popResultSelector(args);
	var _a = argsArgArrayOrObject(args), sources = _a.args, keys = _a.keys;
	var result = new Observable(function(subscriber) {
		var length = sources.length;
		if (!length) {
			subscriber.complete();
			return;
		}
		var values = new Array(length);
		var remainingCompletions = length;
		var remainingEmissions = length;
		var _loop_1 = function(sourceIndex) {
			var hasValue = false;
			innerFrom(sources[sourceIndex]).subscribe(createOperatorSubscriber(subscriber, function(value) {
				if (!hasValue) {
					hasValue = true;
					remainingEmissions--;
				}
				values[sourceIndex] = value;
			}, function() {
				return remainingCompletions--;
			}, void 0, function() {
				if (!remainingCompletions || !hasValue) {
					if (!remainingEmissions) subscriber.next(keys ? createObject(keys, values) : values);
					subscriber.complete();
				}
			}));
		};
		for (var sourceIndex = 0; sourceIndex < length; sourceIndex++) _loop_1(sourceIndex);
	});
	return resultSelector ? result.pipe(mapOneOrManyArgs(resultSelector)) : result;
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/observable/fromEvent.js
var nodeEventEmitterMethods = ["addListener", "removeListener"];
var eventTargetMethods = ["addEventListener", "removeEventListener"];
var jqueryMethods = ["on", "off"];
function fromEvent(target, eventName, options, resultSelector) {
	if (isFunction(options)) {
		resultSelector = options;
		options = void 0;
	}
	if (resultSelector) return fromEvent(target, eventName, options).pipe(mapOneOrManyArgs(resultSelector));
	var _a = __read(isEventTarget(target) ? eventTargetMethods.map(function(methodName) {
		return function(handler) {
			return target[methodName](eventName, handler, options);
		};
	}) : isNodeStyleEventEmitter(target) ? nodeEventEmitterMethods.map(toCommonHandlerRegistry(target, eventName)) : isJQueryStyleEventEmitter(target) ? jqueryMethods.map(toCommonHandlerRegistry(target, eventName)) : [], 2), add = _a[0], remove = _a[1];
	if (!add) {
		if (isArrayLike(target)) return mergeMap(function(subTarget) {
			return fromEvent(subTarget, eventName, options);
		})(innerFrom(target));
	}
	if (!add) throw new TypeError("Invalid event target");
	return new Observable(function(subscriber) {
		var handler = function() {
			var args = [];
			for (var _i = 0; _i < arguments.length; _i++) args[_i] = arguments[_i];
			return subscriber.next(1 < args.length ? args : args[0]);
		};
		add(handler);
		return function() {
			return remove(handler);
		};
	});
}
function toCommonHandlerRegistry(target, eventName) {
	return function(methodName) {
		return function(handler) {
			return target[methodName](eventName, handler);
		};
	};
}
function isNodeStyleEventEmitter(target) {
	return isFunction(target.addListener) && isFunction(target.removeListener);
}
function isJQueryStyleEventEmitter(target) {
	return isFunction(target.on) && isFunction(target.off);
}
function isEventTarget(target) {
	return isFunction(target.addEventListener) && isFunction(target.removeEventListener);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/observable/fromEventPattern.js
function fromEventPattern(addHandler, removeHandler, resultSelector) {
	if (resultSelector) return fromEventPattern(addHandler, removeHandler).pipe(mapOneOrManyArgs(resultSelector));
	return new Observable(function(subscriber) {
		var handler = function() {
			var e = [];
			for (var _i = 0; _i < arguments.length; _i++) e[_i] = arguments[_i];
			return subscriber.next(e.length === 1 ? e[0] : e);
		};
		var retValue = addHandler(handler);
		return isFunction(removeHandler) ? function() {
			return removeHandler(handler, retValue);
		} : void 0;
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/observable/generate.js
function generate(initialStateOrOptions, condition, iterate, resultSelectorOrScheduler, scheduler) {
	var _a, _b;
	var resultSelector;
	var initialState;
	if (arguments.length === 1) _a = initialStateOrOptions, initialState = _a.initialState, condition = _a.condition, iterate = _a.iterate, _b = _a.resultSelector, resultSelector = _b === void 0 ? identity : _b, scheduler = _a.scheduler;
	else {
		initialState = initialStateOrOptions;
		if (!resultSelectorOrScheduler || isScheduler(resultSelectorOrScheduler)) {
			resultSelector = identity;
			scheduler = resultSelectorOrScheduler;
		} else resultSelector = resultSelectorOrScheduler;
	}
	function gen() {
		var state;
		return __generator(this, function(_a) {
			switch (_a.label) {
				case 0:
					state = initialState;
					_a.label = 1;
				case 1:
					if (!(!condition || condition(state))) return [3, 4];
					return [4, resultSelector(state)];
				case 2:
					_a.sent();
					_a.label = 3;
				case 3:
					state = iterate(state);
					return [3, 1];
				case 4: return [2];
			}
		});
	}
	return defer(scheduler ? function() {
		return scheduleIterable(gen(), scheduler);
	} : gen);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/observable/iif.js
function iif(condition, trueResult, falseResult) {
	return defer(function() {
		return condition() ? trueResult : falseResult;
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/observable/timer.js
function timer(dueTime, intervalOrScheduler, scheduler) {
	if (dueTime === void 0) dueTime = 0;
	if (scheduler === void 0) scheduler = async;
	var intervalDuration = -1;
	if (intervalOrScheduler != null) if (isScheduler(intervalOrScheduler)) scheduler = intervalOrScheduler;
	else intervalDuration = intervalOrScheduler;
	return new Observable(function(subscriber) {
		var due = isValidDate(dueTime) ? +dueTime - scheduler.now() : dueTime;
		if (due < 0) due = 0;
		var n = 0;
		return scheduler.schedule(function() {
			if (!subscriber.closed) {
				subscriber.next(n++);
				if (0 <= intervalDuration) this.schedule(void 0, intervalDuration);
				else subscriber.complete();
			}
		}, due);
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/observable/interval.js
function interval(period, scheduler) {
	if (period === void 0) period = 0;
	if (scheduler === void 0) scheduler = asyncScheduler;
	if (period < 0) period = 0;
	return timer(period, period, scheduler);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/observable/merge.js
function merge$1() {
	var args = [];
	for (var _i = 0; _i < arguments.length; _i++) args[_i] = arguments[_i];
	var scheduler = popScheduler(args);
	var concurrent = popNumber(args, Infinity);
	var sources = args;
	return !sources.length ? EMPTY : sources.length === 1 ? innerFrom(sources[0]) : mergeAll(concurrent)(from(sources, scheduler));
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/observable/never.js
var NEVER = new Observable(noop);
function never() {
	return NEVER;
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/argsOrArgArray.js
var isArray = Array.isArray;
function argsOrArgArray(args) {
	return args.length === 1 && isArray(args[0]) ? args[0] : args;
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/observable/onErrorResumeNext.js
function onErrorResumeNext() {
	var sources = [];
	for (var _i = 0; _i < arguments.length; _i++) sources[_i] = arguments[_i];
	var nextSources = argsOrArgArray(sources);
	return new Observable(function(subscriber) {
		var sourceIndex = 0;
		var subscribeNext = function() {
			if (sourceIndex < nextSources.length) {
				var nextSource = void 0;
				try {
					nextSource = innerFrom(nextSources[sourceIndex++]);
				} catch (err) {
					subscribeNext();
					return;
				}
				var innerSubscriber = new OperatorSubscriber(subscriber, void 0, noop, noop);
				nextSource.subscribe(innerSubscriber);
				innerSubscriber.add(subscribeNext);
			} else subscriber.complete();
		};
		subscribeNext();
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/observable/pairs.js
function pairs(obj, scheduler) {
	return from(Object.entries(obj), scheduler);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/util/not.js
function not(pred, thisArg) {
	return function(value, index) {
		return !pred.call(thisArg, value, index);
	};
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/filter.js
function filter(predicate, thisArg) {
	return operate(function(source, subscriber) {
		var index = 0;
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			return predicate.call(thisArg, value, index++) && subscriber.next(value);
		}));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/observable/partition.js
function partition(source, predicate, thisArg) {
	return [filter(predicate, thisArg)(innerFrom(source)), filter(not(predicate, thisArg))(innerFrom(source))];
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/observable/race.js
function race() {
	var sources = [];
	for (var _i = 0; _i < arguments.length; _i++) sources[_i] = arguments[_i];
	sources = argsOrArgArray(sources);
	return sources.length === 1 ? innerFrom(sources[0]) : new Observable(raceInit(sources));
}
function raceInit(sources) {
	return function(subscriber) {
		var subscriptions = [];
		var _loop_1 = function(i) {
			subscriptions.push(innerFrom(sources[i]).subscribe(createOperatorSubscriber(subscriber, function(value) {
				if (subscriptions) {
					for (var s = 0; s < subscriptions.length; s++) s !== i && subscriptions[s].unsubscribe();
					subscriptions = null;
				}
				subscriber.next(value);
			})));
		};
		for (var i = 0; subscriptions && !subscriber.closed && i < sources.length; i++) _loop_1(i);
	};
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/observable/range.js
function range(start, count, scheduler) {
	if (count == null) {
		count = start;
		start = 0;
	}
	if (count <= 0) return EMPTY;
	var end = count + start;
	return new Observable(scheduler ? function(subscriber) {
		var n = start;
		return scheduler.schedule(function() {
			if (n < end) {
				subscriber.next(n++);
				this.schedule();
			} else subscriber.complete();
		});
	} : function(subscriber) {
		var n = start;
		while (n < end && !subscriber.closed) subscriber.next(n++);
		subscriber.complete();
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/observable/using.js
function using(resourceFactory, observableFactory) {
	return new Observable(function(subscriber) {
		var resource = resourceFactory();
		var result = observableFactory(resource);
		(result ? innerFrom(result) : EMPTY).subscribe(subscriber);
		return function() {
			if (resource) resource.unsubscribe();
		};
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/observable/zip.js
function zip$1() {
	var args = [];
	for (var _i = 0; _i < arguments.length; _i++) args[_i] = arguments[_i];
	var resultSelector = popResultSelector(args);
	var sources = argsOrArgArray(args);
	return sources.length ? new Observable(function(subscriber) {
		var buffers = sources.map(function() {
			return [];
		});
		var completed = sources.map(function() {
			return false;
		});
		subscriber.add(function() {
			buffers = completed = null;
		});
		var _loop_1 = function(sourceIndex) {
			innerFrom(sources[sourceIndex]).subscribe(createOperatorSubscriber(subscriber, function(value) {
				buffers[sourceIndex].push(value);
				if (buffers.every(function(buffer) {
					return buffer.length;
				})) {
					var result = buffers.map(function(buffer) {
						return buffer.shift();
					});
					subscriber.next(resultSelector ? resultSelector.apply(void 0, __spreadArray([], __read(result))) : result);
					if (buffers.some(function(buffer, i) {
						return !buffer.length && completed[i];
					})) subscriber.complete();
				}
			}, function() {
				completed[sourceIndex] = true;
				!buffers[sourceIndex].length && subscriber.complete();
			}));
		};
		for (var sourceIndex = 0; !subscriber.closed && sourceIndex < sources.length; sourceIndex++) _loop_1(sourceIndex);
		return function() {
			buffers = completed = null;
		};
	}) : EMPTY;
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/audit.js
function audit(durationSelector) {
	return operate(function(source, subscriber) {
		var hasValue = false;
		var lastValue = null;
		var durationSubscriber = null;
		var isComplete = false;
		var endDuration = function() {
			durationSubscriber === null || durationSubscriber === void 0 || durationSubscriber.unsubscribe();
			durationSubscriber = null;
			if (hasValue) {
				hasValue = false;
				var value = lastValue;
				lastValue = null;
				subscriber.next(value);
			}
			isComplete && subscriber.complete();
		};
		var cleanupDuration = function() {
			durationSubscriber = null;
			isComplete && subscriber.complete();
		};
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			hasValue = true;
			lastValue = value;
			if (!durationSubscriber) innerFrom(durationSelector(value)).subscribe(durationSubscriber = createOperatorSubscriber(subscriber, endDuration, cleanupDuration));
		}, function() {
			isComplete = true;
			(!hasValue || !durationSubscriber || durationSubscriber.closed) && subscriber.complete();
		}));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/auditTime.js
function auditTime(duration, scheduler) {
	if (scheduler === void 0) scheduler = asyncScheduler;
	return audit(function() {
		return timer(duration, scheduler);
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/buffer.js
function buffer(closingNotifier) {
	return operate(function(source, subscriber) {
		var currentBuffer = [];
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			return currentBuffer.push(value);
		}, function() {
			subscriber.next(currentBuffer);
			subscriber.complete();
		}));
		innerFrom(closingNotifier).subscribe(createOperatorSubscriber(subscriber, function() {
			var b = currentBuffer;
			currentBuffer = [];
			subscriber.next(b);
		}, noop));
		return function() {
			currentBuffer = null;
		};
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/bufferCount.js
function bufferCount(bufferSize, startBufferEvery) {
	if (startBufferEvery === void 0) startBufferEvery = null;
	startBufferEvery = startBufferEvery !== null && startBufferEvery !== void 0 ? startBufferEvery : bufferSize;
	return operate(function(source, subscriber) {
		var buffers = [];
		var count = 0;
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			var e_1, _a, e_2, _b;
			var toEmit = null;
			if (count++ % startBufferEvery === 0) buffers.push([]);
			try {
				for (var buffers_1 = __values(buffers), buffers_1_1 = buffers_1.next(); !buffers_1_1.done; buffers_1_1 = buffers_1.next()) {
					var buffer = buffers_1_1.value;
					buffer.push(value);
					if (bufferSize <= buffer.length) {
						toEmit = toEmit !== null && toEmit !== void 0 ? toEmit : [];
						toEmit.push(buffer);
					}
				}
			} catch (e_1_1) {
				e_1 = { error: e_1_1 };
			} finally {
				try {
					if (buffers_1_1 && !buffers_1_1.done && (_a = buffers_1.return)) _a.call(buffers_1);
				} finally {
					if (e_1) throw e_1.error;
				}
			}
			if (toEmit) try {
				for (var toEmit_1 = __values(toEmit), toEmit_1_1 = toEmit_1.next(); !toEmit_1_1.done; toEmit_1_1 = toEmit_1.next()) {
					var buffer = toEmit_1_1.value;
					arrRemove(buffers, buffer);
					subscriber.next(buffer);
				}
			} catch (e_2_1) {
				e_2 = { error: e_2_1 };
			} finally {
				try {
					if (toEmit_1_1 && !toEmit_1_1.done && (_b = toEmit_1.return)) _b.call(toEmit_1);
				} finally {
					if (e_2) throw e_2.error;
				}
			}
		}, function() {
			var e_3, _a;
			try {
				for (var buffers_2 = __values(buffers), buffers_2_1 = buffers_2.next(); !buffers_2_1.done; buffers_2_1 = buffers_2.next()) {
					var buffer = buffers_2_1.value;
					subscriber.next(buffer);
				}
			} catch (e_3_1) {
				e_3 = { error: e_3_1 };
			} finally {
				try {
					if (buffers_2_1 && !buffers_2_1.done && (_a = buffers_2.return)) _a.call(buffers_2);
				} finally {
					if (e_3) throw e_3.error;
				}
			}
			subscriber.complete();
		}, void 0, function() {
			buffers = null;
		}));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/bufferTime.js
function bufferTime(bufferTimeSpan) {
	var _a, _b;
	var otherArgs = [];
	for (var _i = 1; _i < arguments.length; _i++) otherArgs[_i - 1] = arguments[_i];
	var scheduler = (_a = popScheduler(otherArgs)) !== null && _a !== void 0 ? _a : asyncScheduler;
	var bufferCreationInterval = (_b = otherArgs[0]) !== null && _b !== void 0 ? _b : null;
	var maxBufferSize = otherArgs[1] || Infinity;
	return operate(function(source, subscriber) {
		var bufferRecords = [];
		var restartOnEmit = false;
		var emit = function(record) {
			var buffer = record.buffer;
			record.subs.unsubscribe();
			arrRemove(bufferRecords, record);
			subscriber.next(buffer);
			restartOnEmit && startBuffer();
		};
		var startBuffer = function() {
			if (bufferRecords) {
				var subs = new Subscription();
				subscriber.add(subs);
				var record_1 = {
					buffer: [],
					subs
				};
				bufferRecords.push(record_1);
				executeSchedule(subs, scheduler, function() {
					return emit(record_1);
				}, bufferTimeSpan);
			}
		};
		if (bufferCreationInterval !== null && bufferCreationInterval >= 0) executeSchedule(subscriber, scheduler, startBuffer, bufferCreationInterval, true);
		else restartOnEmit = true;
		startBuffer();
		var bufferTimeSubscriber = createOperatorSubscriber(subscriber, function(value) {
			var e_1, _a;
			var recordsCopy = bufferRecords.slice();
			try {
				for (var recordsCopy_1 = __values(recordsCopy), recordsCopy_1_1 = recordsCopy_1.next(); !recordsCopy_1_1.done; recordsCopy_1_1 = recordsCopy_1.next()) {
					var record = recordsCopy_1_1.value;
					var buffer = record.buffer;
					buffer.push(value);
					maxBufferSize <= buffer.length && emit(record);
				}
			} catch (e_1_1) {
				e_1 = { error: e_1_1 };
			} finally {
				try {
					if (recordsCopy_1_1 && !recordsCopy_1_1.done && (_a = recordsCopy_1.return)) _a.call(recordsCopy_1);
				} finally {
					if (e_1) throw e_1.error;
				}
			}
		}, function() {
			while (bufferRecords === null || bufferRecords === void 0 ? void 0 : bufferRecords.length) subscriber.next(bufferRecords.shift().buffer);
			bufferTimeSubscriber === null || bufferTimeSubscriber === void 0 || bufferTimeSubscriber.unsubscribe();
			subscriber.complete();
			subscriber.unsubscribe();
		}, void 0, function() {
			return bufferRecords = null;
		});
		source.subscribe(bufferTimeSubscriber);
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/bufferToggle.js
function bufferToggle(openings, closingSelector) {
	return operate(function(source, subscriber) {
		var buffers = [];
		innerFrom(openings).subscribe(createOperatorSubscriber(subscriber, function(openValue) {
			var buffer = [];
			buffers.push(buffer);
			var closingSubscription = new Subscription();
			var emitBuffer = function() {
				arrRemove(buffers, buffer);
				subscriber.next(buffer);
				closingSubscription.unsubscribe();
			};
			closingSubscription.add(innerFrom(closingSelector(openValue)).subscribe(createOperatorSubscriber(subscriber, emitBuffer, noop)));
		}, noop));
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			var e_1, _a;
			try {
				for (var buffers_1 = __values(buffers), buffers_1_1 = buffers_1.next(); !buffers_1_1.done; buffers_1_1 = buffers_1.next()) buffers_1_1.value.push(value);
			} catch (e_1_1) {
				e_1 = { error: e_1_1 };
			} finally {
				try {
					if (buffers_1_1 && !buffers_1_1.done && (_a = buffers_1.return)) _a.call(buffers_1);
				} finally {
					if (e_1) throw e_1.error;
				}
			}
		}, function() {
			while (buffers.length > 0) subscriber.next(buffers.shift());
			subscriber.complete();
		}));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/bufferWhen.js
function bufferWhen(closingSelector) {
	return operate(function(source, subscriber) {
		var buffer = null;
		var closingSubscriber = null;
		var openBuffer = function() {
			closingSubscriber === null || closingSubscriber === void 0 || closingSubscriber.unsubscribe();
			var b = buffer;
			buffer = [];
			b && subscriber.next(b);
			innerFrom(closingSelector()).subscribe(closingSubscriber = createOperatorSubscriber(subscriber, openBuffer, noop));
		};
		openBuffer();
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			return buffer === null || buffer === void 0 ? void 0 : buffer.push(value);
		}, function() {
			buffer && subscriber.next(buffer);
			subscriber.complete();
		}, void 0, function() {
			return buffer = closingSubscriber = null;
		}));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/catchError.js
function catchError(selector) {
	return operate(function(source, subscriber) {
		var innerSub = null;
		var syncUnsub = false;
		var handledResult;
		innerSub = source.subscribe(createOperatorSubscriber(subscriber, void 0, void 0, function(err) {
			handledResult = innerFrom(selector(err, catchError(selector)(source)));
			if (innerSub) {
				innerSub.unsubscribe();
				innerSub = null;
				handledResult.subscribe(subscriber);
			} else syncUnsub = true;
		}));
		if (syncUnsub) {
			innerSub.unsubscribe();
			innerSub = null;
			handledResult.subscribe(subscriber);
		}
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/scanInternals.js
function scanInternals(accumulator, seed, hasSeed, emitOnNext, emitBeforeComplete) {
	return function(source, subscriber) {
		var hasState = hasSeed;
		var state = seed;
		var index = 0;
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			var i = index++;
			state = hasState ? accumulator(state, value, i) : (hasState = true, value);
			emitOnNext && subscriber.next(state);
		}, emitBeforeComplete && (function() {
			hasState && subscriber.next(state);
			subscriber.complete();
		})));
	};
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/reduce.js
function reduce(accumulator, seed) {
	return operate(scanInternals(accumulator, seed, arguments.length >= 2, false, true));
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/toArray.js
var arrReducer = function(arr, value) {
	return arr.push(value), arr;
};
function toArray() {
	return operate(function(source, subscriber) {
		reduce(arrReducer, [])(source).subscribe(subscriber);
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/joinAllInternals.js
function joinAllInternals(joinFn, project) {
	return pipe(toArray(), mergeMap(function(sources) {
		return joinFn(sources);
	}), project ? mapOneOrManyArgs(project) : identity);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/combineLatestAll.js
function combineLatestAll(project) {
	return joinAllInternals(combineLatest$1, project);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/combineAll.js
var combineAll = combineLatestAll;
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/combineLatest.js
function combineLatest() {
	var args = [];
	for (var _i = 0; _i < arguments.length; _i++) args[_i] = arguments[_i];
	var resultSelector = popResultSelector(args);
	return resultSelector ? pipe(combineLatest.apply(void 0, __spreadArray([], __read(args))), mapOneOrManyArgs(resultSelector)) : operate(function(source, subscriber) {
		combineLatestInit(__spreadArray([source], __read(argsOrArgArray(args))))(subscriber);
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/combineLatestWith.js
function combineLatestWith() {
	var otherSources = [];
	for (var _i = 0; _i < arguments.length; _i++) otherSources[_i] = arguments[_i];
	return combineLatest.apply(void 0, __spreadArray([], __read(otherSources)));
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/concatMap.js
function concatMap(project, resultSelector) {
	return isFunction(resultSelector) ? mergeMap(project, resultSelector, 1) : mergeMap(project, 1);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/concatMapTo.js
function concatMapTo(innerObservable, resultSelector) {
	return isFunction(resultSelector) ? concatMap(function() {
		return innerObservable;
	}, resultSelector) : concatMap(function() {
		return innerObservable;
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/concat.js
function concat() {
	var args = [];
	for (var _i = 0; _i < arguments.length; _i++) args[_i] = arguments[_i];
	var scheduler = popScheduler(args);
	return operate(function(source, subscriber) {
		concatAll()(from(__spreadArray([source], __read(args)), scheduler)).subscribe(subscriber);
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/concatWith.js
function concatWith() {
	var otherSources = [];
	for (var _i = 0; _i < arguments.length; _i++) otherSources[_i] = arguments[_i];
	return concat.apply(void 0, __spreadArray([], __read(otherSources)));
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/observable/fromSubscribable.js
function fromSubscribable(subscribable) {
	return new Observable(function(subscriber) {
		return subscribable.subscribe(subscriber);
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/connect.js
var DEFAULT_CONFIG = { connector: function() {
	return new Subject();
} };
function connect(selector, config) {
	if (config === void 0) config = DEFAULT_CONFIG;
	var connector = config.connector;
	return operate(function(source, subscriber) {
		var subject = connector();
		innerFrom(selector(fromSubscribable(subject))).subscribe(subscriber);
		subscriber.add(source.subscribe(subject));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/count.js
function count(predicate) {
	return reduce(function(total, value, i) {
		return !predicate || predicate(value, i) ? total + 1 : total;
	}, 0);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/debounce.js
function debounce(durationSelector) {
	return operate(function(source, subscriber) {
		var hasValue = false;
		var lastValue = null;
		var durationSubscriber = null;
		var emit = function() {
			durationSubscriber === null || durationSubscriber === void 0 || durationSubscriber.unsubscribe();
			durationSubscriber = null;
			if (hasValue) {
				hasValue = false;
				var value = lastValue;
				lastValue = null;
				subscriber.next(value);
			}
		};
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			durationSubscriber === null || durationSubscriber === void 0 || durationSubscriber.unsubscribe();
			hasValue = true;
			lastValue = value;
			durationSubscriber = createOperatorSubscriber(subscriber, emit, noop);
			innerFrom(durationSelector(value)).subscribe(durationSubscriber);
		}, function() {
			emit();
			subscriber.complete();
		}, void 0, function() {
			lastValue = durationSubscriber = null;
		}));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/debounceTime.js
function debounceTime(dueTime, scheduler) {
	if (scheduler === void 0) scheduler = asyncScheduler;
	return operate(function(source, subscriber) {
		var activeTask = null;
		var lastValue = null;
		var lastTime = null;
		var emit = function() {
			if (activeTask) {
				activeTask.unsubscribe();
				activeTask = null;
				var value = lastValue;
				lastValue = null;
				subscriber.next(value);
			}
		};
		function emitWhenIdle() {
			var targetTime = lastTime + dueTime;
			var now = scheduler.now();
			if (now < targetTime) {
				activeTask = this.schedule(void 0, targetTime - now);
				subscriber.add(activeTask);
				return;
			}
			emit();
		}
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			lastValue = value;
			lastTime = scheduler.now();
			if (!activeTask) {
				activeTask = scheduler.schedule(emitWhenIdle, dueTime);
				subscriber.add(activeTask);
			}
		}, function() {
			emit();
			subscriber.complete();
		}, void 0, function() {
			lastValue = activeTask = null;
		}));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/defaultIfEmpty.js
function defaultIfEmpty(defaultValue) {
	return operate(function(source, subscriber) {
		var hasValue = false;
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			hasValue = true;
			subscriber.next(value);
		}, function() {
			if (!hasValue) subscriber.next(defaultValue);
			subscriber.complete();
		}));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/take.js
function take(count) {
	return count <= 0 ? function() {
		return EMPTY;
	} : operate(function(source, subscriber) {
		var seen = 0;
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			if (++seen <= count) {
				subscriber.next(value);
				if (count <= seen) subscriber.complete();
			}
		}));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/ignoreElements.js
function ignoreElements() {
	return operate(function(source, subscriber) {
		source.subscribe(createOperatorSubscriber(subscriber, noop));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/mapTo.js
function mapTo(value) {
	return map(function() {
		return value;
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/delayWhen.js
function delayWhen(delayDurationSelector, subscriptionDelay) {
	if (subscriptionDelay) return function(source) {
		return concat$1(subscriptionDelay.pipe(take(1), ignoreElements()), source.pipe(delayWhen(delayDurationSelector)));
	};
	return mergeMap(function(value, index) {
		return innerFrom(delayDurationSelector(value, index)).pipe(take(1), mapTo(value));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/delay.js
function delay(due, scheduler) {
	if (scheduler === void 0) scheduler = asyncScheduler;
	var duration = timer(due, scheduler);
	return delayWhen(function() {
		return duration;
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/dematerialize.js
function dematerialize() {
	return operate(function(source, subscriber) {
		source.subscribe(createOperatorSubscriber(subscriber, function(notification) {
			return observeNotification(notification, subscriber);
		}));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/distinct.js
function distinct(keySelector, flushes) {
	return operate(function(source, subscriber) {
		var distinctKeys = /* @__PURE__ */ new Set();
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			var key = keySelector ? keySelector(value) : value;
			if (!distinctKeys.has(key)) {
				distinctKeys.add(key);
				subscriber.next(value);
			}
		}));
		flushes && innerFrom(flushes).subscribe(createOperatorSubscriber(subscriber, function() {
			return distinctKeys.clear();
		}, noop));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/distinctUntilChanged.js
function distinctUntilChanged(comparator, keySelector) {
	if (keySelector === void 0) keySelector = identity;
	comparator = comparator !== null && comparator !== void 0 ? comparator : defaultCompare;
	return operate(function(source, subscriber) {
		var previousKey;
		var first = true;
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			var currentKey = keySelector(value);
			if (first || !comparator(previousKey, currentKey)) {
				first = false;
				previousKey = currentKey;
				subscriber.next(value);
			}
		}));
	});
}
function defaultCompare(a, b) {
	return a === b;
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/distinctUntilKeyChanged.js
function distinctUntilKeyChanged(key, compare) {
	return distinctUntilChanged(function(x, y) {
		return compare ? compare(x[key], y[key]) : x[key] === y[key];
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/throwIfEmpty.js
function throwIfEmpty(errorFactory) {
	if (errorFactory === void 0) errorFactory = defaultErrorFactory;
	return operate(function(source, subscriber) {
		var hasValue = false;
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			hasValue = true;
			subscriber.next(value);
		}, function() {
			return hasValue ? subscriber.complete() : subscriber.error(errorFactory());
		}));
	});
}
function defaultErrorFactory() {
	return new EmptyError();
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/elementAt.js
function elementAt(index, defaultValue) {
	if (index < 0) throw new ArgumentOutOfRangeError();
	var hasDefaultValue = arguments.length >= 2;
	return function(source) {
		return source.pipe(filter(function(v, i) {
			return i === index;
		}), take(1), hasDefaultValue ? defaultIfEmpty(defaultValue) : throwIfEmpty(function() {
			return new ArgumentOutOfRangeError();
		}));
	};
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/endWith.js
function endWith() {
	var values = [];
	for (var _i = 0; _i < arguments.length; _i++) values[_i] = arguments[_i];
	return function(source) {
		return concat$1(source, of.apply(void 0, __spreadArray([], __read(values))));
	};
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/every.js
function every(predicate, thisArg) {
	return operate(function(source, subscriber) {
		var index = 0;
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			if (!predicate.call(thisArg, value, index++, source)) {
				subscriber.next(false);
				subscriber.complete();
			}
		}, function() {
			subscriber.next(true);
			subscriber.complete();
		}));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/exhaustMap.js
function exhaustMap(project, resultSelector) {
	if (resultSelector) return function(source) {
		return source.pipe(exhaustMap(function(a, i) {
			return innerFrom(project(a, i)).pipe(map(function(b, ii) {
				return resultSelector(a, b, i, ii);
			}));
		}));
	};
	return operate(function(source, subscriber) {
		var index = 0;
		var innerSub = null;
		var isComplete = false;
		source.subscribe(createOperatorSubscriber(subscriber, function(outerValue) {
			if (!innerSub) {
				innerSub = createOperatorSubscriber(subscriber, void 0, function() {
					innerSub = null;
					isComplete && subscriber.complete();
				});
				innerFrom(project(outerValue, index++)).subscribe(innerSub);
			}
		}, function() {
			isComplete = true;
			!innerSub && subscriber.complete();
		}));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/exhaustAll.js
function exhaustAll() {
	return exhaustMap(identity);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/exhaust.js
var exhaust = exhaustAll;
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/expand.js
function expand(project, concurrent, scheduler) {
	if (concurrent === void 0) concurrent = Infinity;
	concurrent = (concurrent || 0) < 1 ? Infinity : concurrent;
	return operate(function(source, subscriber) {
		return mergeInternals(source, subscriber, project, concurrent, void 0, true, scheduler);
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/finalize.js
function finalize(callback) {
	return operate(function(source, subscriber) {
		try {
			source.subscribe(subscriber);
		} finally {
			subscriber.add(callback);
		}
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/find.js
function find(predicate, thisArg) {
	return operate(createFind(predicate, thisArg, "value"));
}
function createFind(predicate, thisArg, emit) {
	var findIndex = emit === "index";
	return function(source, subscriber) {
		var index = 0;
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			var i = index++;
			if (predicate.call(thisArg, value, i, source)) {
				subscriber.next(findIndex ? i : value);
				subscriber.complete();
			}
		}, function() {
			subscriber.next(findIndex ? -1 : void 0);
			subscriber.complete();
		}));
	};
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/findIndex.js
function findIndex(predicate, thisArg) {
	return operate(createFind(predicate, thisArg, "index"));
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/first.js
function first(predicate, defaultValue) {
	var hasDefaultValue = arguments.length >= 2;
	return function(source) {
		return source.pipe(predicate ? filter(function(v, i) {
			return predicate(v, i, source);
		}) : identity, take(1), hasDefaultValue ? defaultIfEmpty(defaultValue) : throwIfEmpty(function() {
			return new EmptyError();
		}));
	};
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/groupBy.js
function groupBy(keySelector, elementOrOptions, duration, connector) {
	return operate(function(source, subscriber) {
		var element;
		if (!elementOrOptions || typeof elementOrOptions === "function") element = elementOrOptions;
		else duration = elementOrOptions.duration, element = elementOrOptions.element, connector = elementOrOptions.connector;
		var groups = /* @__PURE__ */ new Map();
		var notify = function(cb) {
			groups.forEach(cb);
			cb(subscriber);
		};
		var handleError = function(err) {
			return notify(function(consumer) {
				return consumer.error(err);
			});
		};
		var activeGroups = 0;
		var teardownAttempted = false;
		var groupBySourceSubscriber = new OperatorSubscriber(subscriber, function(value) {
			try {
				var key_1 = keySelector(value);
				var group_1 = groups.get(key_1);
				if (!group_1) {
					groups.set(key_1, group_1 = connector ? connector() : new Subject());
					var grouped = createGroupedObservable(key_1, group_1);
					subscriber.next(grouped);
					if (duration) {
						var durationSubscriber_1 = createOperatorSubscriber(group_1, function() {
							group_1.complete();
							durationSubscriber_1 === null || durationSubscriber_1 === void 0 || durationSubscriber_1.unsubscribe();
						}, void 0, void 0, function() {
							return groups.delete(key_1);
						});
						groupBySourceSubscriber.add(innerFrom(duration(grouped)).subscribe(durationSubscriber_1));
					}
				}
				group_1.next(element ? element(value) : value);
			} catch (err) {
				handleError(err);
			}
		}, function() {
			return notify(function(consumer) {
				return consumer.complete();
			});
		}, handleError, function() {
			return groups.clear();
		}, function() {
			teardownAttempted = true;
			return activeGroups === 0;
		});
		source.subscribe(groupBySourceSubscriber);
		function createGroupedObservable(key, groupSubject) {
			var result = new Observable(function(groupSubscriber) {
				activeGroups++;
				var innerSub = groupSubject.subscribe(groupSubscriber);
				return function() {
					innerSub.unsubscribe();
					--activeGroups === 0 && teardownAttempted && groupBySourceSubscriber.unsubscribe();
				};
			});
			result.key = key;
			return result;
		}
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/isEmpty.js
function isEmpty() {
	return operate(function(source, subscriber) {
		source.subscribe(createOperatorSubscriber(subscriber, function() {
			subscriber.next(false);
			subscriber.complete();
		}, function() {
			subscriber.next(true);
			subscriber.complete();
		}));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/takeLast.js
function takeLast(count) {
	return count <= 0 ? function() {
		return EMPTY;
	} : operate(function(source, subscriber) {
		var buffer = [];
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			buffer.push(value);
			count < buffer.length && buffer.shift();
		}, function() {
			var e_1, _a;
			try {
				for (var buffer_1 = __values(buffer), buffer_1_1 = buffer_1.next(); !buffer_1_1.done; buffer_1_1 = buffer_1.next()) {
					var value = buffer_1_1.value;
					subscriber.next(value);
				}
			} catch (e_1_1) {
				e_1 = { error: e_1_1 };
			} finally {
				try {
					if (buffer_1_1 && !buffer_1_1.done && (_a = buffer_1.return)) _a.call(buffer_1);
				} finally {
					if (e_1) throw e_1.error;
				}
			}
			subscriber.complete();
		}, void 0, function() {
			buffer = null;
		}));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/last.js
function last(predicate, defaultValue) {
	var hasDefaultValue = arguments.length >= 2;
	return function(source) {
		return source.pipe(predicate ? filter(function(v, i) {
			return predicate(v, i, source);
		}) : identity, takeLast(1), hasDefaultValue ? defaultIfEmpty(defaultValue) : throwIfEmpty(function() {
			return new EmptyError();
		}));
	};
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/materialize.js
function materialize() {
	return operate(function(source, subscriber) {
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			subscriber.next(Notification.createNext(value));
		}, function() {
			subscriber.next(Notification.createComplete());
			subscriber.complete();
		}, function(err) {
			subscriber.next(Notification.createError(err));
			subscriber.complete();
		}));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/max.js
function max(comparer) {
	return reduce(isFunction(comparer) ? function(x, y) {
		return comparer(x, y) > 0 ? x : y;
	} : function(x, y) {
		return x > y ? x : y;
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/flatMap.js
var flatMap = mergeMap;
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/mergeMapTo.js
function mergeMapTo(innerObservable, resultSelector, concurrent) {
	if (concurrent === void 0) concurrent = Infinity;
	if (isFunction(resultSelector)) return mergeMap(function() {
		return innerObservable;
	}, resultSelector, concurrent);
	if (typeof resultSelector === "number") concurrent = resultSelector;
	return mergeMap(function() {
		return innerObservable;
	}, concurrent);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/mergeScan.js
function mergeScan(accumulator, seed, concurrent) {
	if (concurrent === void 0) concurrent = Infinity;
	return operate(function(source, subscriber) {
		var state = seed;
		return mergeInternals(source, subscriber, function(value, index) {
			return accumulator(state, value, index);
		}, concurrent, function(value) {
			state = value;
		}, false, void 0, function() {
			return state = null;
		});
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/merge.js
function merge() {
	var args = [];
	for (var _i = 0; _i < arguments.length; _i++) args[_i] = arguments[_i];
	var scheduler = popScheduler(args);
	var concurrent = popNumber(args, Infinity);
	return operate(function(source, subscriber) {
		mergeAll(concurrent)(from(__spreadArray([source], __read(args)), scheduler)).subscribe(subscriber);
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/mergeWith.js
function mergeWith() {
	var otherSources = [];
	for (var _i = 0; _i < arguments.length; _i++) otherSources[_i] = arguments[_i];
	return merge.apply(void 0, __spreadArray([], __read(otherSources)));
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/min.js
function min(comparer) {
	return reduce(isFunction(comparer) ? function(x, y) {
		return comparer(x, y) < 0 ? x : y;
	} : function(x, y) {
		return x < y ? x : y;
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/multicast.js
function multicast(subjectOrSubjectFactory, selector) {
	var subjectFactory = isFunction(subjectOrSubjectFactory) ? subjectOrSubjectFactory : function() {
		return subjectOrSubjectFactory;
	};
	if (isFunction(selector)) return connect(selector, { connector: subjectFactory });
	return function(source) {
		return new ConnectableObservable(source, subjectFactory);
	};
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/onErrorResumeNextWith.js
function onErrorResumeNextWith() {
	var sources = [];
	for (var _i = 0; _i < arguments.length; _i++) sources[_i] = arguments[_i];
	var nextSources = argsOrArgArray(sources);
	return function(source) {
		return onErrorResumeNext.apply(void 0, __spreadArray([source], __read(nextSources)));
	};
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/pairwise.js
function pairwise() {
	return operate(function(source, subscriber) {
		var prev;
		var hasPrev = false;
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			var p = prev;
			prev = value;
			hasPrev && subscriber.next([p, value]);
			hasPrev = true;
		}));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/pluck.js
function pluck() {
	var properties = [];
	for (var _i = 0; _i < arguments.length; _i++) properties[_i] = arguments[_i];
	var length = properties.length;
	if (length === 0) throw new Error("list of properties cannot be empty.");
	return map(function(x) {
		var currentProp = x;
		for (var i = 0; i < length; i++) {
			var p = currentProp === null || currentProp === void 0 ? void 0 : currentProp[properties[i]];
			if (typeof p !== "undefined") currentProp = p;
			else return;
		}
		return currentProp;
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/publish.js
function publish(selector) {
	return selector ? function(source) {
		return connect(selector)(source);
	} : function(source) {
		return multicast(new Subject())(source);
	};
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/publishBehavior.js
function publishBehavior(initialValue) {
	return function(source) {
		var subject = new BehaviorSubject(initialValue);
		return new ConnectableObservable(source, function() {
			return subject;
		});
	};
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/publishLast.js
function publishLast() {
	return function(source) {
		var subject = new AsyncSubject();
		return new ConnectableObservable(source, function() {
			return subject;
		});
	};
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/publishReplay.js
function publishReplay(bufferSize, windowTime, selectorOrScheduler, timestampProvider) {
	if (selectorOrScheduler && !isFunction(selectorOrScheduler)) timestampProvider = selectorOrScheduler;
	var selector = isFunction(selectorOrScheduler) ? selectorOrScheduler : void 0;
	return function(source) {
		return multicast(new ReplaySubject(bufferSize, windowTime, timestampProvider), selector)(source);
	};
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/raceWith.js
function raceWith() {
	var otherSources = [];
	for (var _i = 0; _i < arguments.length; _i++) otherSources[_i] = arguments[_i];
	return !otherSources.length ? identity : operate(function(source, subscriber) {
		raceInit(__spreadArray([source], __read(otherSources)))(subscriber);
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/repeat.js
function repeat(countOrConfig) {
	var _a;
	var count = Infinity;
	var delay;
	if (countOrConfig != null) if (typeof countOrConfig === "object") _a = countOrConfig.count, count = _a === void 0 ? Infinity : _a, delay = countOrConfig.delay;
	else count = countOrConfig;
	return count <= 0 ? function() {
		return EMPTY;
	} : operate(function(source, subscriber) {
		var soFar = 0;
		var sourceSub;
		var resubscribe = function() {
			sourceSub === null || sourceSub === void 0 || sourceSub.unsubscribe();
			sourceSub = null;
			if (delay != null) {
				var notifier = typeof delay === "number" ? timer(delay) : innerFrom(delay(soFar));
				var notifierSubscriber_1 = createOperatorSubscriber(subscriber, function() {
					notifierSubscriber_1.unsubscribe();
					subscribeToSource();
				});
				notifier.subscribe(notifierSubscriber_1);
			} else subscribeToSource();
		};
		var subscribeToSource = function() {
			var syncUnsub = false;
			sourceSub = source.subscribe(createOperatorSubscriber(subscriber, void 0, function() {
				if (++soFar < count) if (sourceSub) resubscribe();
				else syncUnsub = true;
				else subscriber.complete();
			}));
			if (syncUnsub) resubscribe();
		};
		subscribeToSource();
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/repeatWhen.js
function repeatWhen(notifier) {
	return operate(function(source, subscriber) {
		var innerSub;
		var syncResub = false;
		var completions$;
		var isNotifierComplete = false;
		var isMainComplete = false;
		var checkComplete = function() {
			return isMainComplete && isNotifierComplete && (subscriber.complete(), true);
		};
		var getCompletionSubject = function() {
			if (!completions$) {
				completions$ = new Subject();
				innerFrom(notifier(completions$)).subscribe(createOperatorSubscriber(subscriber, function() {
					if (innerSub) subscribeForRepeatWhen();
					else syncResub = true;
				}, function() {
					isNotifierComplete = true;
					checkComplete();
				}));
			}
			return completions$;
		};
		var subscribeForRepeatWhen = function() {
			isMainComplete = false;
			innerSub = source.subscribe(createOperatorSubscriber(subscriber, void 0, function() {
				isMainComplete = true;
				!checkComplete() && getCompletionSubject().next();
			}));
			if (syncResub) {
				innerSub.unsubscribe();
				innerSub = null;
				syncResub = false;
				subscribeForRepeatWhen();
			}
		};
		subscribeForRepeatWhen();
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/retry.js
function retry(configOrCount) {
	if (configOrCount === void 0) configOrCount = Infinity;
	var config;
	if (configOrCount && typeof configOrCount === "object") config = configOrCount;
	else config = { count: configOrCount };
	var _a = config.count, count = _a === void 0 ? Infinity : _a, delay = config.delay, _b = config.resetOnSuccess, resetOnSuccess = _b === void 0 ? false : _b;
	return count <= 0 ? identity : operate(function(source, subscriber) {
		var soFar = 0;
		var innerSub;
		var subscribeForRetry = function() {
			var syncUnsub = false;
			innerSub = source.subscribe(createOperatorSubscriber(subscriber, function(value) {
				if (resetOnSuccess) soFar = 0;
				subscriber.next(value);
			}, void 0, function(err) {
				if (soFar++ < count) {
					var resub_1 = function() {
						if (innerSub) {
							innerSub.unsubscribe();
							innerSub = null;
							subscribeForRetry();
						} else syncUnsub = true;
					};
					if (delay != null) {
						var notifier = typeof delay === "number" ? timer(delay) : innerFrom(delay(err, soFar));
						var notifierSubscriber_1 = createOperatorSubscriber(subscriber, function() {
							notifierSubscriber_1.unsubscribe();
							resub_1();
						}, function() {
							subscriber.complete();
						});
						notifier.subscribe(notifierSubscriber_1);
					} else resub_1();
				} else subscriber.error(err);
			}));
			if (syncUnsub) {
				innerSub.unsubscribe();
				innerSub = null;
				subscribeForRetry();
			}
		};
		subscribeForRetry();
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/retryWhen.js
function retryWhen(notifier) {
	return operate(function(source, subscriber) {
		var innerSub;
		var syncResub = false;
		var errors$;
		var subscribeForRetryWhen = function() {
			innerSub = source.subscribe(createOperatorSubscriber(subscriber, void 0, void 0, function(err) {
				if (!errors$) {
					errors$ = new Subject();
					innerFrom(notifier(errors$)).subscribe(createOperatorSubscriber(subscriber, function() {
						return innerSub ? subscribeForRetryWhen() : syncResub = true;
					}));
				}
				if (errors$) errors$.next(err);
			}));
			if (syncResub) {
				innerSub.unsubscribe();
				innerSub = null;
				syncResub = false;
				subscribeForRetryWhen();
			}
		};
		subscribeForRetryWhen();
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/sample.js
function sample(notifier) {
	return operate(function(source, subscriber) {
		var hasValue = false;
		var lastValue = null;
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			hasValue = true;
			lastValue = value;
		}));
		innerFrom(notifier).subscribe(createOperatorSubscriber(subscriber, function() {
			if (hasValue) {
				hasValue = false;
				var value = lastValue;
				lastValue = null;
				subscriber.next(value);
			}
		}, noop));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/sampleTime.js
function sampleTime(period, scheduler) {
	if (scheduler === void 0) scheduler = asyncScheduler;
	return sample(interval(period, scheduler));
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/scan.js
function scan(accumulator, seed) {
	return operate(scanInternals(accumulator, seed, arguments.length >= 2, true));
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/sequenceEqual.js
function sequenceEqual(compareTo, comparator) {
	if (comparator === void 0) comparator = function(a, b) {
		return a === b;
	};
	return operate(function(source, subscriber) {
		var aState = createState();
		var bState = createState();
		var emit = function(isEqual) {
			subscriber.next(isEqual);
			subscriber.complete();
		};
		var createSubscriber = function(selfState, otherState) {
			var sequenceEqualSubscriber = createOperatorSubscriber(subscriber, function(a) {
				var buffer = otherState.buffer, complete = otherState.complete;
				if (buffer.length === 0) complete ? emit(false) : selfState.buffer.push(a);
				else !comparator(a, buffer.shift()) && emit(false);
			}, function() {
				selfState.complete = true;
				var complete = otherState.complete, buffer = otherState.buffer;
				complete && emit(buffer.length === 0);
				sequenceEqualSubscriber === null || sequenceEqualSubscriber === void 0 || sequenceEqualSubscriber.unsubscribe();
			});
			return sequenceEqualSubscriber;
		};
		source.subscribe(createSubscriber(aState, bState));
		innerFrom(compareTo).subscribe(createSubscriber(bState, aState));
	});
}
function createState() {
	return {
		buffer: [],
		complete: false
	};
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/share.js
function share(options) {
	if (options === void 0) options = {};
	var _a = options.connector, connector = _a === void 0 ? function() {
		return new Subject();
	} : _a, _b = options.resetOnError, resetOnError = _b === void 0 ? true : _b, _c = options.resetOnComplete, resetOnComplete = _c === void 0 ? true : _c, _d = options.resetOnRefCountZero, resetOnRefCountZero = _d === void 0 ? true : _d;
	return function(wrapperSource) {
		var connection;
		var resetConnection;
		var subject;
		var refCount = 0;
		var hasCompleted = false;
		var hasErrored = false;
		var cancelReset = function() {
			resetConnection === null || resetConnection === void 0 || resetConnection.unsubscribe();
			resetConnection = void 0;
		};
		var reset = function() {
			cancelReset();
			connection = subject = void 0;
			hasCompleted = hasErrored = false;
		};
		var resetAndUnsubscribe = function() {
			var conn = connection;
			reset();
			conn === null || conn === void 0 || conn.unsubscribe();
		};
		return operate(function(source, subscriber) {
			refCount++;
			if (!hasErrored && !hasCompleted) cancelReset();
			var dest = subject = subject !== null && subject !== void 0 ? subject : connector();
			subscriber.add(function() {
				refCount--;
				if (refCount === 0 && !hasErrored && !hasCompleted) resetConnection = handleReset(resetAndUnsubscribe, resetOnRefCountZero);
			});
			dest.subscribe(subscriber);
			if (!connection && refCount > 0) {
				connection = new SafeSubscriber({
					next: function(value) {
						return dest.next(value);
					},
					error: function(err) {
						hasErrored = true;
						cancelReset();
						resetConnection = handleReset(reset, resetOnError, err);
						dest.error(err);
					},
					complete: function() {
						hasCompleted = true;
						cancelReset();
						resetConnection = handleReset(reset, resetOnComplete);
						dest.complete();
					}
				});
				innerFrom(source).subscribe(connection);
			}
		})(wrapperSource);
	};
}
function handleReset(reset, on) {
	var args = [];
	for (var _i = 2; _i < arguments.length; _i++) args[_i - 2] = arguments[_i];
	if (on === true) {
		reset();
		return;
	}
	if (on === false) return;
	var onSubscriber = new SafeSubscriber({ next: function() {
		onSubscriber.unsubscribe();
		reset();
	} });
	return innerFrom(on.apply(void 0, __spreadArray([], __read(args)))).subscribe(onSubscriber);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/shareReplay.js
function shareReplay(configOrBufferSize, windowTime, scheduler) {
	var _a, _b, _c;
	var bufferSize;
	var refCount = false;
	if (configOrBufferSize && typeof configOrBufferSize === "object") _a = configOrBufferSize.bufferSize, bufferSize = _a === void 0 ? Infinity : _a, _b = configOrBufferSize.windowTime, windowTime = _b === void 0 ? Infinity : _b, _c = configOrBufferSize.refCount, refCount = _c === void 0 ? false : _c, scheduler = configOrBufferSize.scheduler;
	else bufferSize = configOrBufferSize !== null && configOrBufferSize !== void 0 ? configOrBufferSize : Infinity;
	return share({
		connector: function() {
			return new ReplaySubject(bufferSize, windowTime, scheduler);
		},
		resetOnError: true,
		resetOnComplete: false,
		resetOnRefCountZero: refCount
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/single.js
function single(predicate) {
	return operate(function(source, subscriber) {
		var hasValue = false;
		var singleValue;
		var seenValue = false;
		var index = 0;
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			seenValue = true;
			if (!predicate || predicate(value, index++, source)) {
				hasValue && subscriber.error(new SequenceError("Too many matching values"));
				hasValue = true;
				singleValue = value;
			}
		}, function() {
			if (hasValue) {
				subscriber.next(singleValue);
				subscriber.complete();
			} else subscriber.error(seenValue ? new NotFoundError("No matching values") : new EmptyError());
		}));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/skip.js
function skip(count) {
	return filter(function(_, index) {
		return count <= index;
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/skipLast.js
function skipLast(skipCount) {
	return skipCount <= 0 ? identity : operate(function(source, subscriber) {
		var ring = new Array(skipCount);
		var seen = 0;
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			var valueIndex = seen++;
			if (valueIndex < skipCount) ring[valueIndex] = value;
			else {
				var index = valueIndex % skipCount;
				var oldValue = ring[index];
				ring[index] = value;
				subscriber.next(oldValue);
			}
		}));
		return function() {
			ring = null;
		};
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/skipUntil.js
function skipUntil(notifier) {
	return operate(function(source, subscriber) {
		var taking = false;
		var skipSubscriber = createOperatorSubscriber(subscriber, function() {
			skipSubscriber === null || skipSubscriber === void 0 || skipSubscriber.unsubscribe();
			taking = true;
		}, noop);
		innerFrom(notifier).subscribe(skipSubscriber);
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			return taking && subscriber.next(value);
		}));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/skipWhile.js
function skipWhile(predicate) {
	return operate(function(source, subscriber) {
		var taking = false;
		var index = 0;
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			return (taking || (taking = !predicate(value, index++))) && subscriber.next(value);
		}));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/startWith.js
function startWith() {
	var values = [];
	for (var _i = 0; _i < arguments.length; _i++) values[_i] = arguments[_i];
	var scheduler = popScheduler(values);
	return operate(function(source, subscriber) {
		(scheduler ? concat$1(values, source, scheduler) : concat$1(values, source)).subscribe(subscriber);
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/switchMap.js
function switchMap(project, resultSelector) {
	return operate(function(source, subscriber) {
		var innerSubscriber = null;
		var index = 0;
		var isComplete = false;
		var checkComplete = function() {
			return isComplete && !innerSubscriber && subscriber.complete();
		};
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			innerSubscriber === null || innerSubscriber === void 0 || innerSubscriber.unsubscribe();
			var innerIndex = 0;
			var outerIndex = index++;
			innerFrom(project(value, outerIndex)).subscribe(innerSubscriber = createOperatorSubscriber(subscriber, function(innerValue) {
				return subscriber.next(resultSelector ? resultSelector(value, innerValue, outerIndex, innerIndex++) : innerValue);
			}, function() {
				innerSubscriber = null;
				checkComplete();
			}));
		}, function() {
			isComplete = true;
			checkComplete();
		}));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/switchAll.js
function switchAll() {
	return switchMap(identity);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/switchMapTo.js
function switchMapTo(innerObservable, resultSelector) {
	return isFunction(resultSelector) ? switchMap(function() {
		return innerObservable;
	}, resultSelector) : switchMap(function() {
		return innerObservable;
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/switchScan.js
function switchScan(accumulator, seed) {
	return operate(function(source, subscriber) {
		var state = seed;
		switchMap(function(value, index) {
			return accumulator(state, value, index);
		}, function(_, innerValue) {
			return state = innerValue, innerValue;
		})(source).subscribe(subscriber);
		return function() {
			state = null;
		};
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/takeUntil.js
function takeUntil(notifier) {
	return operate(function(source, subscriber) {
		innerFrom(notifier).subscribe(createOperatorSubscriber(subscriber, function() {
			return subscriber.complete();
		}, noop));
		!subscriber.closed && source.subscribe(subscriber);
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/takeWhile.js
function takeWhile(predicate, inclusive) {
	if (inclusive === void 0) inclusive = false;
	return operate(function(source, subscriber) {
		var index = 0;
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			var result = predicate(value, index++);
			(result || inclusive) && subscriber.next(value);
			!result && subscriber.complete();
		}));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/tap.js
function tap(observerOrNext, error, complete) {
	var tapObserver = isFunction(observerOrNext) || error || complete ? {
		next: observerOrNext,
		error,
		complete
	} : observerOrNext;
	return tapObserver ? operate(function(source, subscriber) {
		var _a;
		(_a = tapObserver.subscribe) === null || _a === void 0 || _a.call(tapObserver);
		var isUnsub = true;
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			var _a;
			(_a = tapObserver.next) === null || _a === void 0 || _a.call(tapObserver, value);
			subscriber.next(value);
		}, function() {
			var _a;
			isUnsub = false;
			(_a = tapObserver.complete) === null || _a === void 0 || _a.call(tapObserver);
			subscriber.complete();
		}, function(err) {
			var _a;
			isUnsub = false;
			(_a = tapObserver.error) === null || _a === void 0 || _a.call(tapObserver, err);
			subscriber.error(err);
		}, function() {
			var _a, _b;
			if (isUnsub) (_a = tapObserver.unsubscribe) === null || _a === void 0 || _a.call(tapObserver);
			(_b = tapObserver.finalize) === null || _b === void 0 || _b.call(tapObserver);
		}));
	}) : identity;
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/throttle.js
function throttle(durationSelector, config) {
	return operate(function(source, subscriber) {
		var _a = config !== null && config !== void 0 ? config : {}, _b = _a.leading, leading = _b === void 0 ? true : _b, _c = _a.trailing, trailing = _c === void 0 ? false : _c;
		var hasValue = false;
		var sendValue = null;
		var throttled = null;
		var isComplete = false;
		var endThrottling = function() {
			throttled === null || throttled === void 0 || throttled.unsubscribe();
			throttled = null;
			if (trailing) {
				send();
				isComplete && subscriber.complete();
			}
		};
		var cleanupThrottling = function() {
			throttled = null;
			isComplete && subscriber.complete();
		};
		var startThrottle = function(value) {
			return throttled = innerFrom(durationSelector(value)).subscribe(createOperatorSubscriber(subscriber, endThrottling, cleanupThrottling));
		};
		var send = function() {
			if (hasValue) {
				hasValue = false;
				var value = sendValue;
				sendValue = null;
				subscriber.next(value);
				!isComplete && startThrottle(value);
			}
		};
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			hasValue = true;
			sendValue = value;
			!(throttled && !throttled.closed) && (leading ? send() : startThrottle(value));
		}, function() {
			isComplete = true;
			!(trailing && hasValue && throttled && !throttled.closed) && subscriber.complete();
		}));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/throttleTime.js
function throttleTime(duration, scheduler, config) {
	if (scheduler === void 0) scheduler = asyncScheduler;
	var duration$ = timer(duration, scheduler);
	return throttle(function() {
		return duration$;
	}, config);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/timeInterval.js
function timeInterval(scheduler) {
	if (scheduler === void 0) scheduler = asyncScheduler;
	return operate(function(source, subscriber) {
		var last = scheduler.now();
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			var now = scheduler.now();
			var interval = now - last;
			last = now;
			subscriber.next(new TimeInterval(value, interval));
		}));
	});
}
var TimeInterval = function() {
	function TimeInterval(value, interval) {
		this.value = value;
		this.interval = interval;
	}
	return TimeInterval;
}();
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/timeoutWith.js
function timeoutWith(due, withObservable, scheduler) {
	var first;
	var each;
	var _with;
	scheduler = scheduler !== null && scheduler !== void 0 ? scheduler : async;
	if (isValidDate(due)) first = due;
	else if (typeof due === "number") each = due;
	if (withObservable) _with = function() {
		return withObservable;
	};
	else throw new TypeError("No observable provided to switch to");
	if (first == null && each == null) throw new TypeError("No timeout provided.");
	return timeout({
		first,
		each,
		scheduler,
		with: _with
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/timestamp.js
function timestamp(timestampProvider) {
	if (timestampProvider === void 0) timestampProvider = dateTimestampProvider;
	return map(function(value) {
		return {
			value,
			timestamp: timestampProvider.now()
		};
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/window.js
function window(windowBoundaries) {
	return operate(function(source, subscriber) {
		var windowSubject = new Subject();
		subscriber.next(windowSubject.asObservable());
		var errorHandler = function(err) {
			windowSubject.error(err);
			subscriber.error(err);
		};
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			return windowSubject === null || windowSubject === void 0 ? void 0 : windowSubject.next(value);
		}, function() {
			windowSubject.complete();
			subscriber.complete();
		}, errorHandler));
		innerFrom(windowBoundaries).subscribe(createOperatorSubscriber(subscriber, function() {
			windowSubject.complete();
			subscriber.next(windowSubject = new Subject());
		}, noop, errorHandler));
		return function() {
			windowSubject === null || windowSubject === void 0 || windowSubject.unsubscribe();
			windowSubject = null;
		};
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/windowCount.js
function windowCount(windowSize, startWindowEvery) {
	if (startWindowEvery === void 0) startWindowEvery = 0;
	var startEvery = startWindowEvery > 0 ? startWindowEvery : windowSize;
	return operate(function(source, subscriber) {
		var windows = [new Subject()];
		var count = 0;
		subscriber.next(windows[0].asObservable());
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			var e_1, _a;
			try {
				for (var windows_1 = __values(windows), windows_1_1 = windows_1.next(); !windows_1_1.done; windows_1_1 = windows_1.next()) windows_1_1.value.next(value);
			} catch (e_1_1) {
				e_1 = { error: e_1_1 };
			} finally {
				try {
					if (windows_1_1 && !windows_1_1.done && (_a = windows_1.return)) _a.call(windows_1);
				} finally {
					if (e_1) throw e_1.error;
				}
			}
			var c = count - windowSize + 1;
			if (c >= 0 && c % startEvery === 0) windows.shift().complete();
			if (++count % startEvery === 0) {
				var window_2 = new Subject();
				windows.push(window_2);
				subscriber.next(window_2.asObservable());
			}
		}, function() {
			while (windows.length > 0) windows.shift().complete();
			subscriber.complete();
		}, function(err) {
			while (windows.length > 0) windows.shift().error(err);
			subscriber.error(err);
		}, function() {
			windows = null;
		}));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/windowTime.js
function windowTime(windowTimeSpan) {
	var _a, _b;
	var otherArgs = [];
	for (var _i = 1; _i < arguments.length; _i++) otherArgs[_i - 1] = arguments[_i];
	var scheduler = (_a = popScheduler(otherArgs)) !== null && _a !== void 0 ? _a : asyncScheduler;
	var windowCreationInterval = (_b = otherArgs[0]) !== null && _b !== void 0 ? _b : null;
	var maxWindowSize = otherArgs[1] || Infinity;
	return operate(function(source, subscriber) {
		var windowRecords = [];
		var restartOnClose = false;
		var closeWindow = function(record) {
			var window = record.window, subs = record.subs;
			window.complete();
			subs.unsubscribe();
			arrRemove(windowRecords, record);
			restartOnClose && startWindow();
		};
		var startWindow = function() {
			if (windowRecords) {
				var subs = new Subscription();
				subscriber.add(subs);
				var window_1 = new Subject();
				var record_1 = {
					window: window_1,
					subs,
					seen: 0
				};
				windowRecords.push(record_1);
				subscriber.next(window_1.asObservable());
				executeSchedule(subs, scheduler, function() {
					return closeWindow(record_1);
				}, windowTimeSpan);
			}
		};
		if (windowCreationInterval !== null && windowCreationInterval >= 0) executeSchedule(subscriber, scheduler, startWindow, windowCreationInterval, true);
		else restartOnClose = true;
		startWindow();
		var loop = function(cb) {
			return windowRecords.slice().forEach(cb);
		};
		var terminate = function(cb) {
			loop(function(_a) {
				var window = _a.window;
				return cb(window);
			});
			cb(subscriber);
			subscriber.unsubscribe();
		};
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			loop(function(record) {
				record.window.next(value);
				maxWindowSize <= ++record.seen && closeWindow(record);
			});
		}, function() {
			return terminate(function(consumer) {
				return consumer.complete();
			});
		}, function(err) {
			return terminate(function(consumer) {
				return consumer.error(err);
			});
		}));
		return function() {
			windowRecords = null;
		};
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/windowToggle.js
function windowToggle(openings, closingSelector) {
	return operate(function(source, subscriber) {
		var windows = [];
		var handleError = function(err) {
			while (0 < windows.length) windows.shift().error(err);
			subscriber.error(err);
		};
		innerFrom(openings).subscribe(createOperatorSubscriber(subscriber, function(openValue) {
			var window = new Subject();
			windows.push(window);
			var closingSubscription = new Subscription();
			var closeWindow = function() {
				arrRemove(windows, window);
				window.complete();
				closingSubscription.unsubscribe();
			};
			var closingNotifier;
			try {
				closingNotifier = innerFrom(closingSelector(openValue));
			} catch (err) {
				handleError(err);
				return;
			}
			subscriber.next(window.asObservable());
			closingSubscription.add(closingNotifier.subscribe(createOperatorSubscriber(subscriber, closeWindow, noop, handleError)));
		}, noop));
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			var e_1, _a;
			var windowsCopy = windows.slice();
			try {
				for (var windowsCopy_1 = __values(windowsCopy), windowsCopy_1_1 = windowsCopy_1.next(); !windowsCopy_1_1.done; windowsCopy_1_1 = windowsCopy_1.next()) windowsCopy_1_1.value.next(value);
			} catch (e_1_1) {
				e_1 = { error: e_1_1 };
			} finally {
				try {
					if (windowsCopy_1_1 && !windowsCopy_1_1.done && (_a = windowsCopy_1.return)) _a.call(windowsCopy_1);
				} finally {
					if (e_1) throw e_1.error;
				}
			}
		}, function() {
			while (0 < windows.length) windows.shift().complete();
			subscriber.complete();
		}, handleError, function() {
			while (0 < windows.length) windows.shift().unsubscribe();
		}));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/windowWhen.js
function windowWhen(closingSelector) {
	return operate(function(source, subscriber) {
		var window;
		var closingSubscriber;
		var handleError = function(err) {
			window.error(err);
			subscriber.error(err);
		};
		var openWindow = function() {
			closingSubscriber === null || closingSubscriber === void 0 || closingSubscriber.unsubscribe();
			window === null || window === void 0 || window.complete();
			window = new Subject();
			subscriber.next(window.asObservable());
			var closingNotifier;
			try {
				closingNotifier = innerFrom(closingSelector());
			} catch (err) {
				handleError(err);
				return;
			}
			closingNotifier.subscribe(closingSubscriber = createOperatorSubscriber(subscriber, openWindow, openWindow, handleError));
		};
		openWindow();
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			return window.next(value);
		}, function() {
			window.complete();
			subscriber.complete();
		}, handleError, function() {
			closingSubscriber === null || closingSubscriber === void 0 || closingSubscriber.unsubscribe();
			window = null;
		}));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/withLatestFrom.js
function withLatestFrom() {
	var inputs = [];
	for (var _i = 0; _i < arguments.length; _i++) inputs[_i] = arguments[_i];
	var project = popResultSelector(inputs);
	return operate(function(source, subscriber) {
		var len = inputs.length;
		var otherValues = new Array(len);
		var hasValue = inputs.map(function() {
			return false;
		});
		var ready = false;
		var _loop_1 = function(i) {
			innerFrom(inputs[i]).subscribe(createOperatorSubscriber(subscriber, function(value) {
				otherValues[i] = value;
				if (!ready && !hasValue[i]) {
					hasValue[i] = true;
					(ready = hasValue.every(identity)) && (hasValue = null);
				}
			}, noop));
		};
		for (var i = 0; i < len; i++) _loop_1(i);
		source.subscribe(createOperatorSubscriber(subscriber, function(value) {
			if (ready) {
				var values = __spreadArray([value], __read(otherValues));
				subscriber.next(project ? project.apply(void 0, __spreadArray([], __read(values))) : values);
			}
		}));
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/zipAll.js
function zipAll(project) {
	return joinAllInternals(zip$1, project);
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/zip.js
function zip() {
	var sources = [];
	for (var _i = 0; _i < arguments.length; _i++) sources[_i] = arguments[_i];
	return operate(function(source, subscriber) {
		zip$1.apply(void 0, __spreadArray([source], __read(sources))).subscribe(subscriber);
	});
}
//#endregion
//#region node_modules/rxjs/dist/esm5/internal/operators/zipWith.js
function zipWith() {
	var otherInputs = [];
	for (var _i = 0; _i < arguments.length; _i++) otherInputs[_i] = arguments[_i];
	return zip.apply(void 0, __spreadArray([], __read(otherInputs)));
}
//#endregion
export { last as $, ObjectUnsubscribedError as $n, NEVER as $t, scan as A, throwError as An, concatMapTo as At, publishBehavior as B, animationFrame as Bn, bufferTime as Bt, skipUntil as C, ArgumentOutOfRangeError as Cn, take as Ct, shareReplay as D, isObservable as Dn, count as Dt, single as E, EmptyError as En, debounce as Et, repeatWhen as F, observeOn as Fn, toArray as Ft, multicast as G, asyncScheduler as Gn, zip$1 as Gt, pluck as H, queue as Hn, buffer as Ht, repeat as I, EMPTY as In, reduce as It, mergeScan as J, Scheduler as Jn, race as Jt, min as K, asap as Kn, using as Kt, raceWith as L, empty as Ln, catchError as Lt, sample as M, from as Mn, combineLatestWith as Mt, retryWhen as N, scheduled as Nn, combineAll as Nt, share as O, Notification as On, connect as Ot, retry as P, subscribeOn as Pn, combineLatestAll as Pt, materialize as Q, Subject as Qn, onErrorResumeNext as Qt, publishReplay as R, VirtualAction as Rn, bufferWhen as Rt, skipWhile as S, NotFoundError as Sn, ignoreElements as St, skip as T, lastValueFrom as Tn, debounceTime as Tt, pairwise as U, queueScheduler as Un, auditTime as Ut, publish as V, animationFrameScheduler as Vn, bufferCount as Vt, onErrorResumeNextWith as W, async as Wn, audit as Wt, flatMap as X, ReplaySubject as Xn, filter as Xt, mergeMapTo as Y, AsyncSubject as Yn, partition as Yt, max as Z, BehaviorSubject as Zn, pairs as Zt, switchScan as _, bindCallback as _n, distinct as _t, windowToggle as a, generate as an, identity as ar, find as at, switchMap as b, timeout as bn, delayWhen as bt, window as c, forkJoin as cn, noop as cr, exhaust as ct, timeInterval as d, concat$1 as dn, UnsubscriptionError as dr, every as dt, never as en, animationFrames as er, takeLast as et, throttleTime as f, concatAll as fn, endWith as ft, takeUntil as g, bindNodeCallback as gn, distinctUntilChanged as gt, takeWhile as h, combineLatest$1 as hn, distinctUntilKeyChanged as ht, windowWhen as i, iif as in, pipe as ir, findIndex as it, sampleTime as j, of as jn, concatMap as jt, sequenceEqual as k, NotificationKind as kn, concatWith as kt, timestamp as l, connectable as ln, config as lr, exhaustAll as lt, tap as m, mergeMap as mn, throwIfEmpty as mt, zipAll as n, interval as nn, refCount as nr, groupBy as nt, windowTime as o, fromEventPattern as on, observable as or, finalize as ot, throttle as p, mergeAll as pn, elementAt as pt, mergeWith as q, asapScheduler as qn, range as qt, withLatestFrom as r, timer as rn, Observable as rr, first as rt, windowCount as s, fromEvent as sn, Subscriber as sr, expand as st, zipWith as t, merge$1 as tn, ConnectableObservable as tr, isEmpty as tt, timeoutWith as u, defer as un, Subscription as ur, exhaustMap as ut, switchMapTo as v, map as vn, dematerialize as vt, skipLast as w, firstValueFrom as wn, defaultIfEmpty as wt, startWith as x, SequenceError as xn, mapTo as xt, switchAll as y, TimeoutError as yn, delay as yt, publishLast as z, VirtualTimeScheduler as zn, bufferToggle as zt };
